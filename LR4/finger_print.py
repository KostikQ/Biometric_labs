import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

class FingerPrint():
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.fingerprint_gr = cv2.cvtColor(self.fingerprint, cv2.COLOR_BGR2GRAY)
    
    def __call__(self, thres=50, thres_maxval=255, type=cv2.THRESH_BINARY, skel_method='lee', block_width=20, block_height=20):
        self.__preprocessing(thres, thres_maxval, type, skel_method)
        self.nodes, self.tails = self.__get_nodes_and_tails()
        self.n_count, self.t_count = self.__count_nodes_and_tails(block_width, block_height)

    def __preprocessing(self, thresh, maxval, type, method):
        _, self.binary_fingerprint = cv2.threshold(self.fingerprint_gr, thresh=thresh, maxval=maxval, type=type)
        skelet_fingerprint = skeletonize(self.binary_fingerprint, method=method)
        self.skelet_fingerprint = np.array([
            np.array([
                (np.array([
                    255*int(not elem), 255*int(not elem), 255*int(not elem)]))
                for elem in row]) 
            for row in skelet_fingerprint])
    
    def __get_nodes_and_tails(self):
        nodes = []
        tails = []
        for row_ind in range(1, len(self.skelet_fingerprint)-1):
            for col_ind in range(1, len(self.skelet_fingerprint[row_ind])-1):
                if (sum(self.skelet_fingerprint[row_ind][col_ind]) == 0 and 
                    [
                        sum(self.skelet_fingerprint[row_ind - 1][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind -1][col_ind - 1]) == 0, 
                        sum(self.skelet_fingerprint[row_ind + 1][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind + 1][col_ind-1]) == 0,
                        sum(self.skelet_fingerprint[row_ind + 1][col_ind]) == 0, sum(self.skelet_fingerprint[row_ind - 1][col_ind]) == 0, 
                        sum(self.skelet_fingerprint[row_ind][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind][col_ind - 1]) == 0
                     ].count(True) > 2):
                    nodes.append((col_ind, row_ind))
                if (sum(self.skelet_fingerprint[row_ind][col_ind]) == 0 and 
                    [
                        sum(self.skelet_fingerprint[row_ind - 1][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind -1][col_ind - 1]) == 0,
                        sum(self.skelet_fingerprint[row_ind + 1][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind + 1][col_ind-1]) == 0,
                        sum(self.skelet_fingerprint[row_ind + 1][col_ind]) == 0, sum(self.skelet_fingerprint[row_ind - 1][col_ind]) == 0,
                        sum(self.skelet_fingerprint[row_ind][col_ind + 1]) == 0, sum(self.skelet_fingerprint[row_ind][col_ind - 1]) == 0
                    ].count(True) < 2):
                    tails.append([col_ind, row_ind])
        return nodes, tails
    
    def __count_nodes_and_tails(self, block_width, block_height):
        n_count = np.array([])
        t_count = np.array([])
        self.num_block_x = self.fingerprint_gr.shape[1] // block_width
        self.num_block_y = self.fingerprint_gr.shape[0] // block_height
        for row_ind in range(self.num_block_y):
            nodes_in_row = np.array([])
            tails_in_row = np.array([])
            for col_ind in range(self.num_block_x):
                nodes_in_row = np.append(
                    nodes_in_row, sum(
                        1 for x,y in self.nodes 
                        if row_ind * block_height <= y < row_ind * block_height + block_height and 
                        col_ind * block_width <= x < col_ind * block_width + block_width)
                )
                tails_in_row = np.append(
                    tails_in_row, sum(
                        1 for x,y in self.tails 
                        if row_ind * block_height <= y < row_ind * block_height + block_height and 
                        col_ind * block_width <= x < col_ind * block_width + block_width)
                )
            n_count = np.append(n_count, nodes_in_row)
            t_count = np.append(t_count, tails_in_row)
        n_count = n_count.reshape(self.num_block_y, self.num_block_x)
        t_count = t_count.reshape(self.num_block_y, self.num_block_x)
        return n_count, t_count
        
    def draw_detection(self, nodes_color=(255, 0, 0), tails_color=(0, 0, 255)):
        nodes_and_tails = np.copy(self.skelet_fingerprint)
        for node in self.nodes:
            nodes_and_tails = cv2.circle(nodes_and_tails, node, 1, nodes_color, -1)
        for tail in self.tails:
            nodes_and_tails = cv2.circle(nodes_and_tails, tail, 1, tails_color, -1)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.set_figwidth(15)
        fig.set_figheight(10)
        ax[0][0].imshow(self.fingerprint_gr, cmap='gray')
        ax[0][1].imshow(self.binary_fingerprint, cmap='gray')
        ax[1][0].imshow(self.skelet_fingerprint)
        ax[1][1].imshow(nodes_and_tails)
        ax[0][0].set_title('Fingerprint')
        ax[0][1].set_title('Binary fingerprint')
        ax[1][0].set_title('Skeletanization fingerprint')
        ax[1][1].set_title('Nodes and tails')
        ax[0][0].axis(False)
        ax[0][1].axis(False)
        ax[1][0].axis(False)
        ax[1][1].axis(False)
        plt.show()
        fig, ax = plt.subplots(nrows=self.num_block_y, ncols=self.num_block_x)
        for row_ind in range(len(ax)):
            for col_ind in range(len(ax[row_ind])):
                ax[row_ind][col_ind].imshow(nodes_and_tails[row_ind*20:row_ind*20 + 20, col_ind*20:col_ind*20 + 20,:])
                ax[row_ind][col_ind].get_xaxis().set_visible(False)
                ax[row_ind][col_ind].get_yaxis().set_visible(False)
        
        fig.suptitle("Fingerprint's blocks", fontsize=14)
        plt.show()
        plt.close()
     