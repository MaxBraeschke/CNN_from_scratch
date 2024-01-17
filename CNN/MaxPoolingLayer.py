import cupy as np


class MaxPoolingLayer:
    def __init__(self, kernel_size):
        """
        Constructor takes as input the size of the kernel
        """
        self.kernel_size = kernel_size

    def update_w_By(self, d_w):
        return

    def update_b_By(self, d_b):
        return
    
    def forward_prop(self, z_in, a_in,_):
        self.a_in = a_in
        self.batch_size, img_din, img_h, img_w = a_in.shape
        ker_w,ker_h = self.kernel_size, self.kernel_size

        out_h = int(img_h//self.kernel_size)
        out_w = int(img_w//self.kernel_size)

        i0 = np.repeat(np.arange(ker_h), ker_h)
        i1 = np.repeat(np.arange(step=ker_w, start=0 , stop=img_w), out_h)
        j0 = np.tile(np.arange(ker_w), ker_h)
        j1 = np.tile(np.arange(step=ker_h, start=0 , stop=img_h), out_w)

        i=i0.reshape(-1,1)+i1.reshape(1,-1)
        j=j0.reshape(-1,1)+j1.reshape(1,-1)

        select_img = a_in[:,:,i, j]
        
        select_img = select_img.transpose(0,1,3,2)
        
        self.select_img = select_img
        
        a_out = np.max(select_img, axis=(-1)).reshape(self.batch_size,img_din,out_h, out_w)
        
        self.max_idx = np.argmax(select_img, axis=(-1)).reshape(self.batch_size,img_din,out_h,out_w)

        ##should be the same here
        return a_out, a_out

    def getInfo(self):
        return "MP"
    def backward_prop(self, delta,_):
        delta_new = np.zeros(self.a_in.shape)

        b_s, img_din, img_h, img_w = self.a_in.shape
        ker_w, ker_h = self.kernel_size, self.kernel_size
        out_h = int(img_h//self.kernel_size)
        out_w = int(img_w//self.kernel_size)

        y_index_array = np.repeat(np.arange(step=ker_h, start=0 , stop=img_h), out_w).reshape(out_h,out_w)
        y_max_idx = self.max_idx//ker_h
        y_idx = y_max_idx + y_index_array

        x_index_array = np.tile(np.arange(step=ker_w, start=0 , stop=img_w), out_h).reshape(out_h,out_w)
        x_max_idx = self.max_idx%ker_w
        x_idx = x_max_idx + x_index_array

        z_idx = np.tile(np.repeat(np.arange(img_din), out_h*out_w), b_s)
        bs_idx = np.repeat(np.arange(b_s), out_h*out_w*img_din)
        y_max = y_idx.flatten()
        x_max = x_idx.flatten()

        delta_new[bs_idx,z_idx,y_max,x_max] = delta.flatten()

        return delta_new
