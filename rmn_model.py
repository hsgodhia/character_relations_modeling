import torch.nn as nn
import torch.nn.init as init
import torch, copy, random, time, pdb, numpy as np
from torch.autograd import Variable
from util import *
from torch import optim
from itertools import ifilter

class Config(object):
    def __init__(self, compared=[], **kwargs):
        self.name = "rmn"
        self.desc_dim = 30 
        self.book_dim = 50
        self.char_dim = 50
        self.alpha_train_point = 15
        self.train_epochs = 30
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
BATCH_SIZE = 55
span_data, span_size, wmap, cmap, bmap = load_data('data/relationships.csv.gz', 'data/metadata.pkl')
config = Config()
We = cPickle.load(open('data/glove.We', 'rb')).astype('float32')
norm_We = We / np.linalg.norm(We, axis=1)[:, None]
"""
it is basically dividing each word representation by the sqrt(sum(x_i^2))
so we have 16414, 300 divided by 16414, 1 ...one normalizer for each word of the vocabulary
it is basically making it a unit vector for each word, so we are ignoring vector length/magnitude and only relying on 
direction to use it in downstream tasks like similarity calculation and so on
"""
We = np.nan_to_num(norm_We)
We = torch.from_numpy(We)
config.vocab_size, config.emb_dim = We.size(0), We.size(1)
d_word = We.size(1)
num_negs = 50
p_drop = 0.50
config.num_chars = len(cmap)    
config.num_books = len(bmap)
config.vocab_size = len(wmap)
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
        
        self.w_d = nn.Linear(config.desc_dim + config.emb_dim, config.desc_dim, bias=False)
        self.w_rel = nn.Linear(config.desc_dim, config.emb_dim, bias=False)
        
        # the below 3 layers form the transformation from individual span rep to h_in
        self.w_c_to_emb = nn.Linear(config.char_dim, config.emb_dim, bias=False)
        self.w_b_to_emb = nn.Linear(config.book_dim, config.emb_dim, bias=False)
        self.w_vs_to_emb = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
        
        self.v_alpha = nn.Linear(config.emb_dim*2 + config.desc_dim, 1, bias=False)
        self.alpha = Variable(config.alpha_init_val * torch.ones(BATCH_SIZE, 1))
        if torch.cuda.is_available():
            self.alpha = self.alpha.cuda()
        self.train_alpha = False
    
    def set_train_alpha(self, val):
        self.train_alpha = val
        
    def update_alpha(self, input):
        self.alpha = self.sigmoid(self.v_alpha(input))
        #this is batch_size * 1
        
    # the dimension of input is T * B * S where T is the max number of spans available for a given (c1,c2,b) that
    # is considered in a batch B is the batch size and S is the max span size or the
    def forward(self, input):
        bk_id, char_ids, seq, seq_mask, drop_mask, neg_seq, neg_seq_mask, spans_count_l = input        
        v_s = self.w_embed(Variable(seq))        
        seq_mask = seq_mask.unsqueeze(2).expand_as(v_s)
        drop_mask = drop_mask.unsqueeze(2).expand_as(v_s)
        
        v_s_mask = v_s * Variable(seq_mask)
        v_s_dropmask = v_s * Variable(drop_mask)
        
        # v_s has dimension say 8 * 116 * 300
        v_s_mask = torch.sum(v_s_mask, 1).squeeze(1)
        v_s_dropmask = torch.sum(v_s_dropmask, 1).squeeze(1)
        # now v_s is of size (8, 300) one word embedding for each span
        v_s_dropmask = self.w_vs_to_emb(v_s_dropmask)
        
        if neg_seq is not None:
            v_n = self.w_embed(Variable(neg_seq))            
            neg_seq_mask = neg_seq_mask.unsqueeze(2).expand_as(v_n)
            v_n = v_n * Variable(neg_seq_mask)
            v_n = torch.sum(v_n, 1).squeeze(1)
            
        b_emb_in = Variable(bk_id)
        c_emb_in = Variable(char_ids)
        v_b, v_c = self.b_embed(b_emb_in), self.c_embed(c_emb_in)
        # returns vars of size 1*50 and 1*2*50
        c1_var = v_c[:,0,:]
        c2_var = v_c[:,1,:]
        # TODO : no need to add a non linearity to this, simple linear transformation because later we concatenate them and
        # do a non linearity any ways
        v_b, v_c_1, v_c_2 = self.w_b_to_emb(v_b), self.w_c_to_emb(c1_var), self.w_c_to_emb(c2_var)
        # v_c_1 is of size N*300 and v_b of N*300
        v_c = v_c_1 + v_c_2

        if spans_count_l is not None:
            seq_in = torch.zeros(BATCH_SIZE, max(spans_count_l), 300)
            seq_in_dp = torch.zeros(BATCH_SIZE, max(spans_count_l), 300)
            neg_seq_in = torch.zeros(BATCH_SIZE, num_negs, 300)
            if torch.cuda.is_available():
                seq_in = seq_in.cuda()
                seq_in_dp = seq_in_dp.cuda()
                neg_seq_in = neg_seq_in.cuda()
                
            cum_spans_count = 0
            cntr = 0
            for i in spans_count_l:
                # for the original with only sequence mask
                cur_seqq = v_s_mask[cum_spans_count:(cum_spans_count + i), :].data
                pad_res = torch.cat((cur_seqq, torch.zeros(max(spans_count_l) - i, 300).cuda()), 0)
                seq_in[cntr, :, :] = pad_res
                
                # for the original with dropout and sequence mask both
                cur_seqq_dp = v_s_dropmask[cum_spans_count:(cum_spans_count + i), :].data
                pad_res_dp = torch.cat((cur_seqq, torch.zeros(max(spans_count_l) - i, 300).cuda()), 0)
                seq_in_dp[cntr, :, :] = pad_res_dp
                
                if neg_seq is not None:
                    neg_seq_in[cntr,:,:] = v_n[cntr*num_negs:(cntr + 1)*num_negs, :].data
                cum_spans_count += i
                cntr += 1
            if neg_seq is not None:
                del v_n
        del v_s
        # initalize

        seq_in = Variable(seq_in)
        seq_in_dp = Variable(seq_in_dp)
        if neg_seq is not None:
            neg_seq_in = Variable(neg_seq_in)
            
        d_t = Variable(torch.zeros(BATCH_SIZE, config.desc_dim))
        total_loss = Variable(torch.zeros(1,))
        zrs = Variable(torch.zeros(BATCH_SIZE, num_negs))

        if torch.cuda.is_available():
            d_t = d_t.cuda()
            total_loss = total_loss.cuda()
            zrs = zrs.cuda()        
            
        trajn = []
        for t in range(max(spans_count_l)):
            # the dropout one is used here to calculate the mixed span representation
            v_st_dp = seq_in_dp[:, t, :]
            
            # the default only seq mask and no dropout applied is used to calculate the loss
            v_s_mask = seq_in[:, t, :]
            
            # 20 * 300
            h_in = v_st_dp + v_b + v_c
            h_t = self.relu(h_in)
            
            # size is 1 * 300
            if self.train_alpha:
                self.update_alpha(torch.cat((h_t, d_t, v_st), 1))
            
            # scalars like 1 is automatically broadcasted into a vector to do element wise subtraction
            d_t = self.alpha.expand_as(d_t) * (self.softmax(self.w_d(torch.cat((h_t, d_t), 1)))) + (1 - self.alpha.expand_as(d_t))*d_t
            # d_t = self.softmax(self.w_d(torch.cat((h_t, d_t), 1)))
            # because i'm getting a continuous sequence I removed the direct carry forward from previous to see how it works
            # size is 1 * 30 the number of descriptors, a weighting over it
            
            # save the relationship state for each time step and return it as the trajectory for the given data point
            # each data point corresponds to a single character pair and book and all spans of it
            if config.eval:
                trajn.append(d_t.data.cpu()) # move it out of gpu memory
            if neg_seq is None:
                continue
            
            # this is the reconstruction vector made using the dictionary and the hidden state vector d_t
            r_t = self.w_rel(d_t)            
            
            # normalization here
            r_t = r_t / torch.norm(r_t, 2, 1).expand_as(r_t)
            
            v_s_mask = v_s_mask / torch.norm(v_s_mask, 2, 1).expand_as(v_s_mask)            
            v_s_mask[v_s_mask != v_s_mask] = 0
            
            neg_seq_in = neg_seq_in / torch.norm(neg_seq_in, 2, 2).expand_as(neg_seq_in)
            neg_seq_in[neg_seq_in != neg_seq_in] = 0
            
            # this is the negative loss in the max margin equation
            v_n_res = torch.bmm(neg_seq_in, r_t.unsqueeze(2)).squeeze(2)
            
            recon_loss = torch.bmm(r_t.unsqueeze(1), v_s_mask.unsqueeze(2))
            recon_loss = recon_loss.squeeze(2)
            recon_loss = recon_loss.repeat(1, num_negs)
            
            cur_loss = torch.sum(torch.max(zrs, 1 - recon_loss + v_n_res), 1)
            #pdb.set_trace()
            # this is batch_size * 1
            # this mask is for removing data points which dont have a valid value for this time step
            mask = Variable(torch.from_numpy((t < np.array(spans_count_l)).astype('float')).float()).cuda()
            loss = torch.dot(cur_loss, mask)
            total_loss += loss
            

        # enforce orthogonality constraint
        w_rel_mat = self.w_rel.weight.data
        # w_rel is a weight matrix of size d * K so we want to normalize each of the K descriptors along the 0 axis
        norm_vec = torch.norm(w_rel_mat, 2, 0)
        norm_vec = norm_vec.expand_as(w_rel_mat)
        w_rel_mat_unit = w_rel_mat / norm_vec  # d * K time K * d is d * d
        ortho_penalty = 1e-6 * torch.sum((torch.dot(w_rel_mat_unit.t(), w_rel_mat_unit) - torch.eye(w_rel_mat_unit.size(1))) ** 2)
        total_loss += ortho_penalty
        
        del seq_in, seq_in_dp, neg_seq_in, d_t, drop_mask, seq_mask, zrs
        # if you want to return multiple things put them into a list else it throws an error
        return total_loss, trajn

def weights_init(m):
    if isinstance(m, nn.Linear):
        #xavier needs fan-in and fan-out to be defined, which means rows and cols, so weight atleast 2D
        init.xavier_uniform(m.weight.data)
        if m.bias:
            init.uniform(m.bias.data) 

def train_epoch(mdl, optimizer):
    random.shuffle(span_data)
    losses, bk_l, ch_l, curr_l, cm_l, dp_l,  ns_l, nm_l, num_spans = [], [], [], [], [], [], [], [], []
    batch_cnt = 0
    for book, chars, curr, cm in span_data:
        ns, nm = generate_negative_samples(num_traj, span_size, num_negs, span_data)

        book = torch.from_numpy(book).long()
        chars = torch.from_numpy(chars).long().view(1, 2)
        curr = torch.from_numpy(curr).long()
        ns = torch.from_numpy(ns).long()

        cm = torch.from_numpy(cm)
        nm = torch.from_numpy(nm)
        # word dropout
        # randomly generates numbers in the interval from [0,1) 
        drop_mask = (torch.rand(*(cm.size())) < (1 - p_drop)).float()
        drop_mask = drop_mask * cm

        if torch.cuda.is_available():
            book = book.cuda()
            chars = chars.cuda()
            curr = curr.cuda()
            cm = cm.cuda()
            drop_mask = drop_mask.cuda()
            ns = ns.cuda()
            nm = nm.cuda()

            bk_l.append(book)
            ch_l.append(chars)
            curr_l.append(curr)
            num_spans.append(curr.size(0))
            cm_l.append(cm)
            dp_l.append(drop_mask)
            ns_l.append(ns)
            nm_l.append(nm)

            batch_cnt += 1
            if batch_cnt % BATCH_SIZE == 0:
                batch_cnt = 0
                bk_in = torch.cat(bk_l)
                ch_in = torch.cat(ch_l)
                curr_in = torch.cat(curr_l)
                cm_in = torch.cat(cm_l)
                ns_in = torch.cat(ns_l)
                nm_in = torch.cat(nm_l)
                dp_in = torch.cat(dp_l)
                # call training function here to get cost and loss                            
                optimizer.zero_grad()
                #if mdl.train_alpha:
                #alpha_optim.zero_grad()               
                loss, _ = mdl([bk_in, ch_in, curr_in, cm_in, dp_in, ns_in, nm_in, num_spans])
                #print(loss.data[0])
                losses.append(loss.data[0])
                loss.backward()
                optimizer.step()                
                #if mdl.train_alpha:
                #alpha_optim.step()

                del bk_l[:], ch_l[:], curr_l[:], cm_l[:], dp_l[:], ns_l[:], nm_l[:], num_spans[:]
                del bk_in, ch_in, curr_in, cm_in, ns_in, nm_in, dp_in
                
    if len(num_spans) > 0:
        # process the remaining element which were not the % BATCH SIZE
        global BATCH_SIZE
        BATCH_SIZE = len(num_spans)
        #print(mdl.alpha)
        mdl.alpha = mdl.alpha[:BATCH_SIZE]

        bk_in = torch.cat(bk_l)
        ch_in = torch.cat(ch_l)
        curr_in = torch.cat(curr_l)
        cm_in = torch.cat(cm_l)
        ns_in = torch.cat(ns_l)
        nm_in = torch.cat(nm_l)
        dp_in = torch.cat(dp_l)

        # call training function here to get cost and loss                            
        optimizer.zero_grad()
        loss, _ = mdl([bk_in, ch_in, curr_in, cm_in, dp_in, ns_in, nm_in, num_spans])
        #print(loss.data[0])
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()                

        del bk_l[:], ch_l[:], curr_l[:], cm_l[:], dp_l[:], ns_l[:], nm_l[:], num_spans[:]
        
    return sum(losses) / len(span_data)
        
def train(n_epochs):
    print d_word, span_size, config.desc_dim, config.vocab_size, config.num_chars, config.num_books, num_traj
    print 'compiling...'
    # build neural network here
    mdl = RMNModel(config, We)
    if torch.cuda.is_available():
        mdl.cuda()
    mdl.apply(weights_init)
    
    params = filter(lambda p: p.requires_grad, mdl.parameters())
    optimizer = optim.Adam(params)
    print 'done compiling, now training...'

    for epoch in range(n_epochs):
        if epoch >= config.alpha_train_point:
            mdl.set_train_alpha(True)
            mdl.w_rel.weight.requires_grad = False
        start_time = time.time()    
        eloss = train_epoch(mdl, optimizer)
        end_time = time.time()
        print 'done with epoch: ', epoch, ' cost =', eloss, 'time: ', end_time-start_time
        
        global BATCH_SIZE
        BATCH_SIZE = 55
        mdl.alpha = mdl.alpha[0, :].repeat(BATCH_SIZE, 1)
                              
    torch.save(mdl.state_dict(), "model_13.pth")

"""
Since the descriptors are represented in the same 300 dimension space as that of the vocabulary
we can find nearest neighbors of the descriptor vector and select a label from the 10 most similar vocab words
"""
def save_descriptors(descriptor_log, weight_mat, We, revmap):
    We = We.numpy()
    # original weight matrix is emb_dim * desc_dim
    print 'writing descriptors...'
    R = weight_mat.cpu().t().numpy() #this is of desc_dim * emb_dim
    log = open(descriptor_log, 'w')
    for ind in range(len(R)):
        desc = R[ind] / np.linalg.norm(R[ind])
        sims = We.dot(desc.T)
        ordered_words = np.argsort(sims)[::-1]
        desc_list = [ revmap[w] for w in ordered_words[:10]]
        log.write(' '.join(desc_list) + '\n')
        print('descriptor %d:' % ind)
        print(desc_list)
    log.flush()
    log.close()

def save_trajectories(trajectory_log, span_data, bmap, cmap, mdl):
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
        if bname != "B0036CIQWS":
            continue
        if c1 != "Joe" and c2 != "Joe":
            continue
        book = torch.from_numpy(book).long()
        chars = torch.from_numpy(chars).long().unsqueeze(0)

        curr = torch.from_numpy(curr).long()
        cm = torch.from_numpy(cm)
        drop_mask = (torch.rand(*(cm.size())) < (1 - p_drop)).float()
            
        if torch.cuda.is_available():
            book = book.cuda()
            chars = chars.cuda()
            curr = curr.cuda()
            cm = cm.cuda()
            drop_mask = drop_mask.cuda()
            
        _, traj = mdl([book, chars, curr, cm, drop_mask, None, None, [cm.size(0)]])
        print("{} {} {} {}".format(bname, c1, c2, len(traj)))
        for ind in range(len(traj)):
            step = traj[ind].squeeze(0)
            traj_writer.writerow([bname, c1, c2, ind, step.numpy().tolist()])
        bc += 1
        if bc > 10:
            break
    tlog.flush()
    tlog.close()

def test():    
    global BATCH_SIZE
    BATCH_SIZE = 1
    print 'loading data...'
    descriptor_log = 'descriptors_model_13.log'
    trajectory_log = 'trajectories_13.log'        
    print d_word, span_size, config.desc_dim, config.vocab_size, config.num_chars, config.num_books, num_traj
    config.eval = True
    mdl = RMNModel(config, We)
    if torch.cuda.is_available():
        mdl.cuda()        
    saved_state = torch.load("model_13.pth")
    mdl.load_state_dict(saved_state)
    mdl.eval()
    save_trajectories(trajectory_log, span_data, bmap, cmap, mdl)    
    #save_descriptors(descriptor_log, mdl.w_rel.weight.data, We, revmap)

#train(config.train_epochs)
#test()
