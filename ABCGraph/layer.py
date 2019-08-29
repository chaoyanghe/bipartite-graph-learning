from abcgraph_adv import ABCGraphAdversarial

# TODO: This class implements the layer-wise function of ABCGraph
class Layer(ABCGraphAdversarial):

    def __init__(self):
        super(Layer, self).__init__()


# explicit training
logging.info('###Depth 1 starts!\n')
for i in range(self.epochs):
    for iter in range(self.batch_num_u):
        start_index = self.batch_size * iter
        end_index = self.batch_size * (iter + 1)
        if iter == self.batch_num_u - 1:
            end_index = self.u_num
        u_attr_batch = self.u_attr[start_index:end_index]
        u_adj_batch = self.u_adj[start_index:end_index]

        # prepare data to the tensor
        u_attr_tensor = torch.as_tensor(u_attr_batch, dtype=torch.float, device=self.device)
        u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

        # training
        gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
        lossD, lossG = self.adversarial_explicit.forward_backward(u_attr_tensor, gcn_explicit_output, step=1,
                                                                  epoch=i,
                                                                  iter=iter)
        self.f_loss.write("%s %s\n" % (lossD, lossG))
        # wandb.log({"lossD": lossD, "epoch": i + self.epochs*0})
        # wandb.log({"lossG": lossG, "epoch": i + self.epochs*0})


# explicit inference
u_explicit_attr = torch.FloatTensor([]).to(self.device)
for iter in range(self.batch_num_u):
    start_index = self.batch_size * iter
    end_index = self.batch_size * (iter + 1)
    if iter == self.batch_num_u - 1:
        end_index = self.u_num
    u_adj_batch = self.u_adj[start_index:end_index]

    # prepare data to the tensor
    u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

    # inference
    gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
    u_explicit_attr = torch.cat((u_explicit_attr, gcn_explicit_output.detach()), 0)