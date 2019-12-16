from skimage.util.shape import view_as_blocks
import numpy as np

A = np.random.rand(4,4)
print(A)
print('*' * 100)
print(A[0,0], A[0,1], A[1,0])

# B = view_as_blocks(A, block_shape=(2, 2,3))
# print('Block shape: ',B.shape)

# print(B)

def blocks(image, cell_size=[2, 2]):
    h = image.shape[0]
    w = image.shape[1]
    cell_h = h//cell_size[0]
    cell_w = w // cell_size[1]
    cells = np.zeros((cell_h, cell_w))
     
    for row in range(0, h, cell_h):
        for col in range(0, w, cell_w):
            cell = image[row:row + cell_h, col:col + cell_h]
            # cell = cell.reshape((1, 12))
            # cell = np.squeeze(cell)
            max_el = np.average(cell)
            print('****** (', str(row), ',', str(col), ')', cell, 'maax: ', max_el)
            
            image[row:row + cell_h, col:col + cell_h] = max_el
    # print(cells.shape)
    print(image)
    threshold = 0.5
    result = (image > threshold)*1
    # image[idx] = 1
    print('mask*'*20)
    print(result)

blocks(A)
