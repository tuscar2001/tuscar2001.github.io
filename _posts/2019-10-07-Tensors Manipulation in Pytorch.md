---
layout: post
title:  "Tensors Manipulation in Pytorch"
date:   2019-10-07 11:21:15 
---

I am showing below how to use Pytorch for your common manipulations. Pytorch is a powerful tool for AI and is pretty easy to use. It is very close to numpy and can be learned very fast if you already know numpy


```python
import torch
import numpy as np
```

At the time this blog is written, we are using the following Pytorch version


```python
print ("Pytorch version",torch.__version__)
```

    Pytorch version 1.2.0
    

<h2>Initializations</h2>

We first start by creating a tensor array. The tensor below has two rows and two dimensions


```python
tensor_arr = torch.Tensor([[1,8],[9,5]])
print (tensor_array.shape)
```

    torch.Size([2, 2])
    

Tensor is a dataflow that passes through the computational nodes of your graph.


```python
tensor_arr
```




    tensor([[1., 8.],
            [9., 5.]])



Here is how to initialize a tensor. 


```python
uninitialized_tens = torch.Tensor(3,3)
```

This is equivalent to np.zeros in numpy


```python
uninitialized_tens
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])



Here is how to get the number of elements in a tensor.


```python
torch.numel(tensor_uninitialized)
```




    9



Initialized tensor


```python
tensor_init = torch.rand(4, 5)
print (tensor_init)
```

    tensor([[0.9728, 0.1311, 0.6161, 0.0662, 0.1314],
            [0.0479, 0.9058, 0.3572, 0.3595, 0.3990],
            [0.1130, 0.8729, 0.5676, 0.6363, 0.6682],
            [0.3813, 0.6013, 0.1263, 0.5421, 0.6029]])
    

You can specify the type of your tensor just as numpy arrays


```python
tensor_integer = torch.randn(5,3).type(torch.IntTensor)
tensor_integer
```




    tensor([[ 0, -1,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  3,  0],
            [-1, -1,  0]], dtype=torch.int32)




```python
Here is a float data converted to long type
```


```python
tensor_long = torch.LongTensor([1.0,2.0,3.0]) 
```

LongTensors are heavily used in Deep Learning when minimizing the Cross Entropy Loss. For more information about tensor types, please visit the following link:https://jdhao.github.io/2017/11/15/pytorch-datatype-note/

<h2>Basic Operations</h2>

You can initialize a tensor of ones just like in numpy


```python
tensor_ones = torch.ones(10)
tensor_ones
```




    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])



You can also do the same with zeros  just like in numpy


```python
tensor_zero = torch.zeros(10)
tensor_zero
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



You can buid an identity matrix from Pytorch


```python
tensor_id = torch.eye(5)
tensor_id
```




    tensor([[1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]])



From there, you can filter your tensor on non zeros by returnin the non zero indices


```python
no_zero = torch.nonzero(tensor_id)
no_zero
```




    tensor([[0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]])



What if you wanted to call a matrix of zeros on a preexisting tensor?


```python
tensor_zeromat = torch.zeros_like(tensor_id)
tensor_zeromat
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])



You can do it similarly with ones


```python
tensor_onemat = torch.ones_like(tensor_id)
tensor_onemat
```




    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])



Initialize a tensor with random numbers


```python
tensor_init = torch.rand(5,5)
tensor_init
```




    tensor([[0.6364, 0.4855, 0.9646, 0.3216, 0.7987],
            [0.6469, 0.5354, 0.8046, 0.5328, 0.8189],
            [0.5826, 0.8617, 0.3659, 0.7883, 0.8722],
            [0.7728, 0.1201, 0.0641, 0.5286, 0.3470],
            [0.1882, 0.6031, 0.0157, 0.2448, 0.2829]])



the inplace parameter (equivalent to inplace = True in numpy) is called in pytorch using "_" after the method as below add_ operation is another example.


```python
tensor_init.fill_(7)
```




    tensor([[7., 7., 7., 7., 7.],
            [7., 7., 7., 7., 7.],
            [7., 7., 7., 7., 7.],
            [7., 7., 7., 7., 7.],
            [7., 7., 7., 7., 7.]])



Let's make this a little bit more interesting by converting our tensor to numpy arrays and vice versa


```python
myList = range(5)
arr = np.array(myList)
arr
```




    array([0, 1, 2, 3, 4])



Converting a numpy array into a tensor is done the following way


```python
torchTens = torch.from_numpy(arr)
torchTens
```




    tensor([0, 1, 2, 3, 4], dtype=torch.int32)



It is easy to convert back a tensor to a numpy array


```python
tens2np = torchTens.numpy()
tens2np
```




    array([0, 1, 2, 3, 4])



<h2>Slicing</h2>

**Beware!!!** A tensor created from a numpy array share the same memory in that array. Evey change in value in the array is therefore reflected in the tensor


```python
arr[3]
```




    3




```python
arr[3]=100
```


```python
print(arr)
print (tens2np)
```

    [  0   1   2 100   4]
    [  0   1   2 100   4]
    

If you are familiar with slicing and element retrieval in array, you are then comfortable with tensors


```python
torchTens2 = torch.randn(5,6)
torchTens2
```




    tensor([[-1.9801, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926],
            [ 2.3804, -0.2179, -0.5940,  2.2083, -0.0774,  1.7591],
            [-0.3347, -0.1207,  0.4339, -0.4425,  1.7139,  1.7440],
            [-1.1394,  2.0804,  0.4753,  0.0719, -0.6040,  0.9928],
            [-2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155]])



retrieving the second row and 3rd column element from the tensor


```python
torchTens2[1,2]
```




    tensor(-0.5940)



retrieving all the rows of the third column


```python
torchTens2[:,2]
```




    tensor([ 1.6706, -0.5940,  0.4339,  0.4753,  0.8593])



retrieving all the rows of the last two columns


```python
torchTens2[:,-2:]
```




    tensor([[-1.3473,  0.6926],
            [-0.0774,  1.7591],
            [ 1.7139,  1.7440],
            [-0.6040,  0.9928],
            [-1.0395, -1.4155]])



<h2>Dimensions</h2>

Getting the shape of the tensor can be done in two ways: .shape or .size()


```python
print (torchTens2.size())
```

    torch.Size([5, 6])
    


```python
print (torchTens2.shape)
```

    torch.Size([5, 6])
    

<h2>Views</h2>

Just like numpy arrays, tensors can be viewed in different ways. Just make sure the view parameter(s) have the total size of the tensor. In this very case, the view will expect 30 


```python
tensorReshaped = torchTens2.view(30)
tensorReshaped
```




    tensor([-1.9801, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926,  2.3804, -0.2179,
            -0.5940,  2.2083, -0.0774,  1.7591, -0.3347, -0.1207,  0.4339, -0.4425,
             1.7139,  1.7440, -1.1394,  2.0804,  0.4753,  0.0719, -0.6040,  0.9928,
            -2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155])



This is a 1-D tensor. You know this by the number of brackets in the output below


```python
tensorReshaped.shape
```




    torch.Size([30])



**Beware!!!:** Each update in the original tensor is also reflected in the view


```python
torchTens2[0,0] = 5
```


```python
print (torchTens2)
```

    tensor([[ 5.0000, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926],
            [ 2.3804, -0.2179, -0.5940,  2.2083, -0.0774,  1.7591],
            [-0.3347, -0.1207,  0.4339, -0.4425,  1.7139,  1.7440],
            [-1.1394,  2.0804,  0.4753,  0.0719, -0.6040,  0.9928],
            [-2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155]])
    


```python
print (tensorReshaped)
```

    tensor([ 5.0000, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926,  2.3804, -0.2179,
            -0.5940,  2.2083, -0.0774,  1.7591, -0.3347, -0.1207,  0.4339, -0.4425,
             1.7139,  1.7440, -1.1394,  2.0804,  0.4753,  0.0719, -0.6040,  0.9928,
            -2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155])
    

You can specify part of the dimension of your tensor by focusing either on the rows or the columns

row focus: I am sure I want ten rows, but I am not concerned about the number of columns


```python
tensorRowShape = torchTens2.view(10,-1)
```


```python
tensorRowShape
```




    tensor([[ 5.0000, -0.8114,  1.6706],
            [ 0.4205, -1.3473,  0.6926],
            [ 2.3804, -0.2179, -0.5940],
            [ 2.2083, -0.0774,  1.7591],
            [-0.3347, -0.1207,  0.4339],
            [-0.4425,  1.7139,  1.7440],
            [-1.1394,  2.0804,  0.4753],
            [ 0.0719, -0.6040,  0.9928],
            [-2.0198, -1.0512,  0.8593],
            [-1.2236, -1.0395, -1.4155]])



Column focus: I am sure I want 15 columns, but I am not concerned about the number of rows


```python
tensorColShape = torchTens2.view(-1,15)
```


```python
tensorColShape
```




    tensor([[ 5.0000, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926,  2.3804, -0.2179,
             -0.5940,  2.2083, -0.0774,  1.7591, -0.3347, -0.1207,  0.4339],
            [-0.4425,  1.7139,  1.7440, -1.1394,  2.0804,  0.4753,  0.0719, -0.6040,
              0.9928, -2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155]])



No need to mention that tensorRowShape and tensorColShape are both 2-D tensors because of the double brackets

<h2>Sorting</h2>

Sorting tensor returns the sorted values as well as the sorted indices. Sorting is done by default for each row in ascending order.


```python
sortTensor, sortIndices = torch.sort(torchTens2)
```


```python
print (sortTensor)
```

    tensor([[-1.3473, -0.8114,  0.4205,  0.6926,  1.6706,  5.0000],
            [-0.5940, -0.2179, -0.0774,  1.7591,  2.2083,  2.3804],
            [-0.4425, -0.3347, -0.1207,  0.4339,  1.7139,  1.7440],
            [-1.1394, -0.6040,  0.0719,  0.4753,  0.9928,  2.0804],
            [-2.0198, -1.4155, -1.2236, -1.0512, -1.0395,  0.8593]])
    


```python
print (sortIndices)
```

    tensor([[4, 1, 3, 5, 2, 0],
            [2, 1, 4, 5, 3, 0],
            [3, 0, 1, 2, 4, 5],
            [0, 4, 3, 2, 5, 1],
            [0, 5, 3, 1, 4, 2]])
    

However, we can specify how we want the sorting to be made. Rowise (dim = 1) vs columnwise (dim = 0)


```python
torchTens2
```




    tensor([[ 5.0000, -0.8114,  1.6706,  0.4205, -1.3473,  0.6926],
            [ 2.3804, -0.2179, -0.5940,  2.2083, -0.0774,  1.7591],
            [-0.3347, -0.1207,  0.4339, -0.4425,  1.7139,  1.7440],
            [-1.1394,  2.0804,  0.4753,  0.0719, -0.6040,  0.9928],
            [-2.0198, -1.0512,  0.8593, -1.2236, -1.0395, -1.4155]])




```python
sortTensorCol, sortIndicesCol = torch.sort(torchTens2, dim = 0)
```


```python
sortTensorCol
```




    tensor([[-2.0198, -1.0512, -0.5940, -1.2236, -1.3473, -1.4155],
            [-1.1394, -0.8114,  0.4339, -0.4425, -1.0395,  0.6926],
            [-0.3347, -0.2179,  0.4753,  0.0719, -0.6040,  0.9928],
            [ 2.3804, -0.1207,  0.8593,  0.4205, -0.0774,  1.7440],
            [ 5.0000,  2.0804,  1.6706,  2.2083,  1.7139,  1.7591]])




```python
sortIndicesCol
```




    tensor([[4, 4, 1, 4, 0, 4],
            [3, 0, 2, 2, 4, 0],
            [2, 1, 3, 3, 3, 3],
            [1, 2, 4, 0, 1, 2],
            [0, 3, 0, 1, 2, 1]])




```python
sortTensorRow, sortIndicesRow = torch.sort(torchTens2, dim = 1)
```


```python
sortTensorRow
```




    tensor([[-1.3473, -0.8114,  0.4205,  0.6926,  1.6706,  5.0000],
            [-0.5940, -0.2179, -0.0774,  1.7591,  2.2083,  2.3804],
            [-0.4425, -0.3347, -0.1207,  0.4339,  1.7139,  1.7440],
            [-1.1394, -0.6040,  0.0719,  0.4753,  0.9928,  2.0804],
            [-2.0198, -1.4155, -1.2236, -1.0512, -1.0395,  0.8593]])




```python
sortIndicesRow
```




    tensor([[4, 1, 3, 5, 2, 0],
            [2, 1, 4, 5, 3, 0],
            [3, 0, 1, 2, 4, 5],
            [0, 4, 3, 2, 5, 1],
            [0, 5, 3, 1, 4, 2]])




```python
With these manipulations, you should now be able to use Pytorch for your AI models.
```

<p>
Please check out this website also if interested in <a href="https://www.realestatedipdive.com/locations/florida/" rel="nofollow">Land Investment</a>
</p>