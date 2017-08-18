import torch.nn as nn
import torch.nn.init as init
import torch, copy, random, time, pdb, numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from util import *
from torch import optim
from itertools import ifilter

class Config(object):
    def __init__(self, compared=[], **kwargs):
        self.name = "rmn"
        self.word_drop = 0.75
        self.desc_dim = 30
        self.book_dim = 50
        self.num_negs = 50
        self.char_dim = 50
        self.alpha_train_point = 15
        self.train_epochs = 20
        self.alpha_init_val = 0.5
        self.eval = False
        self.vocb_size = None
        self.emb_dim = None       
        self.num_books = None
        self.num_chars = None
        
    def __repr__(self):
        ks = sorted(k for k in self.__dict__ if k not in ['name'])
        return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)

# ---global data and config initializations---
BATCH_SIZE, prc_batch_cn = 50, 0
span_data, span_size, wmap, cmap, bmap = load_data('data/relationships.csv.gz', 'data/metadata.pkl')
config = Config()
"""
it is basically dividing each word representation by the sqrt(sum(x_i^2))
so we have 16414, 300 divided by 16414, 1 ...one normalizer for each word of the vocabulary
it is basically making it a unit vector for each word, so we are ignoring vector length/magnitude and only relying on 
direction to use it in downstream tasks like similarity calculation and so on
"""
We = cPickle.load(open('data/glove.We', 'rb')).astype('float32')
We = torch.from_numpy(We)
We = F.normalize(We)

config.vocab_size, config.emb_dim, d_word = We.size(0), We.size(1), We.size(1)

config.num_chars = len(cmap)    
config.num_books = len(bmap)
config.vocab_size = len(wmap)

# this is basically one data point, where it is in turn composed of multiple time steps or spans
num_traj = len(span_data)
revmap = {}
for w in wmap:
    revmap[wmap[w]] = w
# ---initialization close

class RMNModel(nn.Module):        
    def __init__(self, config, emb_data):
        super(RMNModel, self).__init__()
        # the embedding layer to lookup the pre-trained glove embedding of span words
        self.w_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.w_embed.weight.requires_grad = False
        self.w_embed.weight.data.copy_(emb_data)
        
        self.c_embed = nn.Embedding(config.num_chars, config.char_dim)
        self.b_embed = nn.Embedding(config.num_books, config.book_dim)

        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.word_drop)
        
        self.w_d_h = nn.Linear(config.emb_dim, config.desc_dim, bias=False)
        self.w_d_prev = nn.Linear(config.desc_dim, config.desc_dim, bias=False)
        self.w_rel = nn.Linear(config.desc_dim, config.emb_dim, bias=False)
        
        # the below 3 layers form the transformation from individual span rep to h_in
        self.w_c_to_emb = nn.Linear(config.char_dim, config.emb_dim, bias=False)
        self.w_b_to_emb = nn.Linear(config.book_dim, config.emb_dim, bias=False)
        self.w_vs_to_emb = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
        
        self.v_alpha = nn.Linear(config.emb_dim*2 + config.desc_dim, 1, bias=False)
        self.alpha = Variable(config.alpha_init_val * torch.ones(1,), requires_grad=False)
        if torch.cuda.is_available():
            self.alpha = self.alpha.cuda()
        self.train_alpha = False
        
    def set_train_alpha(self, val):
        self.train_alpha = val
        
    def update_alpha(self, input):
        self.alpha = self.sigmoid(self.v_alpha(Variable(input.data)))
        #this is batch_size * 1        
        
    # the dimension of input is T * B * S where T is the max number of spans available for a given (c1,c2,b) that
    # is considered in a batch B is the batch size and S is the max span size or the
    def forward(self, input):
        # seq is size N * M where N = batch size and M = max sequence length
        bk_id, char_ids, seq, seq_mask, neg_seq, neg_seq_mask, spans_count_l = input                
        drop_mask = self.dropout(seq_mask)
        if self.training:
            drop_mask = drop_mask * (1 - config.word_drop)
        # v_s has dimension say 8 * 116 * 300
        # is of size N * M * 300
        v_s = self.w_embed(seq)

        temp_ones = Variable(torch.ones(drop_mask.size(0), 1)).cuda()

        # mean out the sequence dimension
        seq_mask = seq_mask.unsqueeze(2)
        v_s_mask = v_s * seq_mask
        seq_mask_sums = torch.sum(seq_mask, 1)
        seq_mask_sums = torch.max(seq_mask_sums, temp_ones)        
        v_s_mask = torch.sum(v_s_mask, 1) / seq_mask_sums
            
        drop_mask = drop_mask.unsqueeze(2)
        drop_mask_sums = torch.sum(drop_mask, 1)
        drop_mask_sums = torch.max(drop_mask_sums, temp_ones)
        
        v_s_dropmask = v_s * drop_mask
        v_s_dropmask = torch.sum(v_s_dropmask, 1) / drop_mask_sums
        v_s_dropmask = self.w_vs_to_emb(v_s_dropmask)
        # now v_s is of size (8, 300) one word embedding for each span
        
        
        if neg_seq is not None:
            v_n = self.w_embed(neg_seq)
            #the negative words are not dropped out
            neg_seq_mask = neg_seq_mask.unsqueeze(2)
            v_n = v_n * neg_seq_mask
            v_n = torch.sum(v_n, 1) / torch.sum(neg_seq_mask, 1)
            
        v_b, v_c = self.b_embed(bk_id), self.c_embed(char_ids)
        # returns vars of size 1*50 and 1*2*50
        c1_var = v_c[:,0,:]
        c2_var = v_c[:,1,:]
        v_b, v_c_1, v_c_2 = self.w_b_to_emb(v_b), self.w_c_to_emb(c1_var), self.w_c_to_emb(c2_var)
        # v_c_1 is of size N*300 and v_b of N*300
        v_c = v_c_1 + v_c_2
        
        if spans_count_l is not None:
            # the second dimension is basically storing the maximum number of time steps that we can have for any data point
            seq_in = Variable(torch.zeros(BATCH_SIZE, max(spans_count_l), 300))
            seq_in_dp = Variable(torch.zeros(BATCH_SIZE, max(spans_count_l), 300))
            neg_seq_in = Variable(torch.zeros(BATCH_SIZE, config.num_negs, 300))

            if torch.cuda.is_available():
                seq_in = seq_in.cuda()
                seq_in_dp = seq_in_dp.cuda()
                neg_seq_in = neg_seq_in.cuda()
                
            cum_spans_count = 0
            cntr = 0
            for i in spans_count_l:
                # for the original with only sequence mask
                cur_seqq = v_s_mask[cum_spans_count:(cum_spans_count + i), :]
                if i != max(spans_count_l):
                    pad_res = torch.cat((cur_seqq, Variable(torch.zeros(max(spans_count_l) - i, 300)).cuda()), 0)
                    seq_in[cntr, :, :] = pad_res
                else:
                    seq_in[cntr, :, :] = cur_seqq
                
                # for the original with dropout and sequence mask both
                cur_seqq_dp = v_s_dropmask[cum_spans_count:(cum_spans_count + i), :]
                if i != max(spans_count_l):
                    pad_res_dp = torch.cat((cur_seqq_dp, Variable(torch.zeros(max(spans_count_l) - i, 300)).cuda()), 0)         
                    seq_in_dp[cntr, :, :] = pad_res_dp
                else:
                    seq_in_dp[cntr, :, :] = cur_seqq_dp
                
                if neg_seq is not None:
                    neg_seq_in[cntr,:,:] = v_n[cntr*config.num_negs:(cntr + 1)*config.num_negs, :]
                cum_spans_count += i
                cntr += 1
            if neg_seq is not None:
                del v_n
        del v_s

        # initalize
        total_loss = 0
        prev_d_t = Variable(torch.zeros(BATCH_SIZE, config.desc_dim), requires_grad=False)
        zrs = Variable(torch.zeros(BATCH_SIZE, config.num_negs), requires_grad=False)
        if torch.cuda.is_available():
            zrs = zrs.cuda()        
            prev_d_t = prev_d_t.cuda()
            
        trajn = []
        # compute the d_t vectors in parallel
        for t in range(max(spans_count_l)):
            # the dropout one is used here to calculate the mixed span representation
            v_st_dp = seq_in_dp[:, t, :].detach()            
            # the default only seq mask and no dropout applied is used to calculate the loss
            v_st_mask = seq_in[:, t, :].detach()
            
            # 20 * 300
            h_in = v_st_dp + v_b + v_c
            h_t = self.relu(h_in)     
            d_t = self.alpha * self.softmax(self.w_d_h(h_t) + self.w_d_prev(prev_d_t))  + (1 - self.alpha) * prev_d_t
            # dt is of size batch_Size * 30
            sv = np.sum(np.isnan(d_t.data.cpu().numpy()).astype(int))
            if sv > 0:
                #pdb.set_trace()
                print("got nan in d_t")

            # size is 1 * 300
            if self.train_alpha:                
                self.update_alpha(torch.cat((h_t, d_t, v_st_dp), 1))
                sv2 = np.sum(np.isnan(self.alpha.data.cpu().numpy()).astype(int))
                if sv2 > 0:
                    print("got nan in alpha")
                    #pdb.set_trace()
            # save the relationship state for each time step and return it as the trajectory for the given data point
            # each data point corresponds to a single character pair and book and all spans of it
            if config.eval:
                trajn.append(d_t.data.cpu()) # move it out of gpu memory
            if neg_seq is None:
                continue
            
            # this is the reconstruction vector made using the dictionary and the hidden state vector d_t
            r_t = self.w_rel(d_t) # is of size BATCH * 300
            # normalization here            
            r_t = F.normalize(r_t) 
            v_st_mask = F.normalize(v_st_mask) # default is euclidean along the dim=1
            neg_seq_in = F.normalize(neg_seq_in, 2, 2) # default eps is 1e-12
            
            # this is the negative loss in the max margin equation
            # BATCH_SIZE * NUM_NEG * 300 times BATCH_SIZE * 1 * 300
            #v_n_res = torch.bmm(neg_seq_in, r_t.unsqueeze(2)).squeeze(2)
            v_n_res = neg_seq_in * r_t.unsqueeze(1)
            v_n_res = torch.sum(v_n_res, 2)
            # BATCH_SIZE * NUM_NEG
            
            # each of these is a matrix of size BATCH_SIZE * 300
            # we are doing a similarity between the two vectors like a dot product
            recon_loss = r_t * v_st_mask
            recon_loss = torch.sum(recon_loss, 1, keepdim=True)
            # now the recon loss is of size BATCH_SIZE * 1
                
            cur_loss = torch.sum(torch.max(zrs, 1 - recon_loss + v_n_res), 1)
            # this is batch_size * 1
            # this mask is for removing data points which dont have a valid value for this time step
            mask = Variable(torch.from_numpy((t < np.array(spans_count_l)).astype('float')).float()).cuda()
            loss = torch.dot(cur_loss, mask)
            total_loss += loss
            
            prev_d_t = d_t

        w_rel_mat = self.w_rel.weight
        # w_rel is a weight matrix of size d * K so we want to normalize each of the K descriptors along the 0 axis

        w_rel_mat_unit = F.normalize(w_rel_mat, 2, 0)
        w_rel_mm = torch.mm(w_rel_mat_unit.t(), w_rel_mat_unit)
        id_mat = Variable(torch.eye(w_rel_mat_unit.size(1))).cuda()
        w_rel_mm = w_rel_mm.sub(id_mat)
        
        ortho_penalty = 1e-6 * torch.norm(w_rel_mm)
        if total_loss is not None:
            total_loss += ortho_penalty
        del seq_in, seq_in_dp, neg_seq_in, prev_d_t, d_t, seq_mask, zrs
        # if you want to return multiple things put them into a list else it throws an error
        return total_loss, trajn


def train_epoch(mdl, optimizer):
    random.shuffle(span_data)
    losses, bk_l, ch_l, curr_l, cm_l, dp_l,  ns_l, nm_l, num_spans = [], [], [], [], [], [], [], [], []
    prc_batch_cn, batch_cnt = 0, 0
    #temp_data = span_data[:200]
    for book, chars, curr, cm in span_data:
        # for each relation with s spans we generate n negative spans
        ns, nm = generate_negative_samples(num_traj, span_size, config.num_negs, span_data)
        book = torch.from_numpy(book).long()
        chars = torch.from_numpy(chars).long().view(1, 2)
        curr = torch.from_numpy(curr).long()
        ns = torch.from_numpy(ns).long()

        cm = torch.from_numpy(cm)
        nm = torch.from_numpy(nm)
        # word dropout

        if torch.cuda.is_available():
            book = book.cuda() # one book
            chars = chars.cuda() # one pair of character
            curr = curr.cuda() # list of spans for the above relation
            cm = cm.cuda() # the sequence mask for each span 
            ns = ns.cuda()
            nm = nm.cuda()

        bk_l.append(book)
        ch_l.append(chars)
        curr_l.append(curr)
        num_spans.append(curr.size(0))
        cm_l.append(cm)
        ns_l.append(ns)
        nm_l.append(nm)

        batch_cnt += 1
        if batch_cnt % BATCH_SIZE == 0:
            batch_cnt = 0
            bk_in = Variable(torch.cat(bk_l))
            ch_in = Variable(torch.cat(ch_l))
            curr_in = Variable(torch.cat(curr_l))
            cm_in = Variable(torch.cat(cm_l))
            ns_in = Variable(torch.cat(ns_l))
            nm_in = Variable(torch.cat(nm_l))

            # call training function here to get cost and loss                            
            optimizer.zero_grad()           
            loss, _ = mdl([bk_in, ch_in, curr_in, cm_in, ns_in, nm_in, num_spans])
            prc_batch_cn += 1
            losses.append(loss.data[0])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(mdl.parameters(), 10)
            optimizer.step()                

            del bk_l[:], ch_l[:], curr_l[:], cm_l[:], ns_l[:], nm_l[:], num_spans[:]
            del bk_in, ch_in, curr_in, cm_in, ns_in, nm_in
                
    if len(num_spans) > 0:
        # process the remaining element which were not the % BATCH SIZE
        global BATCH_SIZE
        BATCH_SIZE = len(num_spans)
        mdl.alpha = mdl.alpha[0].repeat(BATCH_SIZE, 1)
        
        bk_in = Variable(torch.cat(bk_l))
        ch_in = Variable(torch.cat(ch_l))
        curr_in = Variable(torch.cat(curr_l))
        cm_in = Variable(torch.cat(cm_l))
        ns_in = Variable(torch.cat(ns_l))
        nm_in = Variable(torch.cat(nm_l))
            
        # call training function here to get cost and loss                            
        optimizer.zero_grad()
        loss, _ = mdl([bk_in, ch_in, curr_in, cm_in, ns_in, nm_in, num_spans])
        
        prc_batch_cn += 1
        losses.append(loss.data[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(mdl.parameters(), 10)
        optimizer.step()                

        del bk_l[:], ch_l[:], curr_l[:], cm_l[:], ns_l[:], nm_l[:], num_spans[:]
        
    return sum(losses) / len(span_data)
        
def train(n_epochs):
    print d_word, span_size, config.desc_dim, config.vocab_size, config.num_chars, config.num_books, num_traj
    print 'compiling...'
    
    # build neural network here
    mdl = RMNModel(config, We)
    
    # enter train mode
    mdl.train()
    
    # transfer to gpu
    if torch.cuda.is_available():
        mdl.cuda()
        
    # print parameters and initialize them here
    for name, p in mdl.named_parameters():
        print(name, p.size(), p.requires_grad, type(p))
        if name == 'c_embed.weight' or name == 'b_embed.weight':
            print('init', name)
            init.normal(p)
        elif name == 'w_embed.weight':
            continue
        elif 'bias' not in name:
            print('init', name)
            init.xavier_uniform(p)
        else:
            print('init', name)
            init.constant(p, 0)
                
    params = list(filter(lambda p: p.requires_grad, mdl.parameters()))
    print('total params', len(params))
    optimizer = optim.Adam(params)
    print 'done compiling, now training...'

    min_loss = None
    for epoch in range(n_epochs):
        if epoch >= config.alpha_train_point:
            mdl.set_train_alpha(True)
            mdl.w_rel.weight.requires_grad = False
        start_time = time.time()    
        eloss = train_epoch(mdl, optimizer)
        end_time = time.time()
        print 'done with epoch: ', epoch, ' cost =', eloss, 'time: ', end_time - start_time
        if min_loss is None or eloss < min_loss:
            torch.save(mdl.state_dict(), "model_16.pth")
            torch.save(optimizer.state_dict(), "optimizer_16.pth")
        global BATCH_SIZE
        BATCH_SIZE = 50
        mdl.alpha = mdl.alpha[0].repeat(BATCH_SIZE, 1)
        
    torch.save(mdl.state_dict(), "model_16_last.pth")

"""
Since the descriptors are represented in the same 300 dimension space as that of the vocabulary
we can find nearest neighbors of the descriptor vector and select a label from the 10 most similar vocab words
"""
def save_descriptors(descriptor_log, weight_mat, We, revmap):
    We = We.numpy()
    # original weight matrix is emb_dim * desc_dim
    print 'writing descriptors...'
    R = F.normalize(weight_mat, 2, 0).cpu().numpy() # now this is of emb_dim * desc_dim

    log = open(descriptor_log, 'w')
    for ind in range(R.shape[1]):
        desc = R[:,ind]        
        # We is vocab * 300
        sims = We.dot(desc)
        # this is a short cut way to reverse the array [::-1]
        ordered_words = np.argsort(sims)[::-1]
        desc_list = [ revmap[w] for w in ordered_words[:10]]
        log.write(' '.join(desc_list) + '\n')
        print('descriptor %d:' % ind)
        print(desc_list)
    log.flush()
    log.close()

def save_trajectories(trajectory_log, span_data, bmap, cmap, mdl):
    potter_books = ['B019PIOJYU', 'B019PIOJY0', 'B019PIOJVI', 'B019PIOJV8', 'B019PIOJZE', 'B019PIOJZ4', 'B019PIOJWW']
    print 'writing trajectories...'
    tlog = open(trajectory_log, 'wb')
    traj_writer = csv.writer(tlog)
    traj_writer.writerow(['Book', 'Char 1', 'Char 2', 'Span ID'] + \
        ['Topic ' + str(i) for i in range(30)])
    bc = 0
    print(len(span_data))
    for book, chars, curr, cm in span_data:
        c1, c2 = [cmap[c] for c in chars]
        bname = bmap[book[0]]
        if bname != 'Dracula' and bname != 'BourneBetrayal' and bname != 'RisingTides' and bname != 'BourneDeception':
            continue
        if c1 != 'Arthur' and c2 != 'Arthur':
            continue
        book = torch.from_numpy(book).long()
        chars = torch.from_numpy(chars).long().unsqueeze(0)

        curr = torch.from_numpy(curr).long()
        cm = torch.from_numpy(cm)
            
        if torch.cuda.is_available():
            book = Variable(book).cuda()
            chars = Variable(chars).cuda()
            curr = Variable(curr).cuda()
            cm = Variable(cm).cuda()
            
        _, traj = mdl([book, chars, curr, cm, None, None, [cm.size(0)]])
        print("{} {} {} {}".format(bname, c1, c2, len(traj)))
        for ind in range(len(traj)):
            step = traj[ind].squeeze(0)
            traj_writer.writerow([bname, c1, c2, ind, step.numpy().tolist()])
        bc += 1
        if bc > 5:
            break
    tlog.flush()
    tlog.close()

def test():    
    global BATCH_SIZE
    BATCH_SIZE = 1
    print 'loading data...'
    descriptor_log = 'descriptors_model_16.log'
    trajectory_log = 'trajectories_16.log'        
    print d_word, span_size, config.desc_dim, config.vocab_size, config.num_chars, config.num_books, num_traj
    config.eval = True
    mdl = RMNModel(config, We)
    if torch.cuda.is_available():
        mdl.cuda()        
    saved_state = torch.load("model_16.pth")
    mdl.load_state_dict(saved_state)
    mdl.eval()
    #save_trajectories(trajectory_log, span_data, bmap, cmap, mdl)    
    save_descriptors(descriptor_log, mdl.w_rel.weight.data, We, revmap)

if __name__ == '__main__':
    train(config.train_epochs)
    #test()