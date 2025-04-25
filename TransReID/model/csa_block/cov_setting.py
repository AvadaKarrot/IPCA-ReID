import torch

def make_cov_index_matrix(dim):  # make symmetric matrix for embedding index
    matrix = torch.LongTensor()
    s_index = 0
    for i in range(dim):
        matrix = torch.cat([matrix, torch.arange(s_index, s_index + dim).unsqueeze(0)], dim=0)
        s_index += (dim - (2 + i))
    return matrix.triu(diagonal=1).transpose(0, 1) + matrix.triu(diagonal=1)


class CovMatrix_AIAW:

    # 初始化类的基本参数和变量。
    def __init__(self, dim, relax_denom=0):
        """
            dim (int): 协方差矩阵的维度。
            relax_denom (int, optional): 控制考虑的非对角线条目的数量放松参数。默认为0。
        """

        super(CovMatrix_AIAW, self).__init__()
        self.dim = dim # 768
        self.i = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda() # 上三角矩阵（不含对角线）

        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = torch.sum(self.reversal_i)# 非对角线元素的数量
        self.num_sensitive = 0 # 敏感条目的数量
        self.cov_matrix = None #  # 协方差矩阵
        self.count_pair_cov = 0 # 配对协方差计数
        self.mask_matrix = None # 掩码矩阵
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self): # 返回单位矩阵和上三角矩阵。
        # Returns the identity matrix and its transposed version, used for calculations.
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True): # 返回单位矩阵、掩码矩阵、敏感条目的数量和移除的协方差信息。如果掩码矩阵尚未设置，会调用 set_mask_matrix 方法来设置。
        #  returns the identity matrix, the mask matrix, the number of sensitive entries, and the number of removed covariances.
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self): # 计算并设置掩码矩阵，选择最敏感的协方差元素，并将其标记为1。根据计算的协方差矩阵来更新掩码矩阵。
        # torch.set_printoptions(threshold=500000)
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)

        if self.margin == 0:    
            # num_sensitive = int(4/1000 * cov_flatten.size()[0])
            num_sensitive = int(0/10000 * cov_flatten.size()[0])
            print('cov_flatten.size()[0]', cov_flatten.size()[0])
            print("num_sensitive =", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1 # 设置最敏感元素的掩码为 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)

    def set_pair_covariance(self, pair_cov):
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            self.cov_matrix = self.cov_matrix + pair_cov
            ################## debug ##################
            # print('batch pair shape is {}'.format(pair_cov.shape))
            # print(pair_cov.device)
            # print('cov_matix shape is {}'.format(self.cov_matrix.shape))
            # print(self.cov_matrix.device)
            ################## debug ##################
        self.count_pair_cov += 1
