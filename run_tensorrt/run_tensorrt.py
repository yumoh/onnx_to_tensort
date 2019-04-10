import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class Engine:
    """
    input a tensorrt engine file.then use to inference
    """
    def __init__(self,tensorrt_engine_path:str):
        assert(os.path.exists(tensorrt_engine_path))
        with open(tensorrt_engine_path, "rb") as fp:
            self.runtime = trt.Runtime(trt.Logger()) 
            self.engine = self.runtime.deserialize_cuda_engine(fp.read())
        # 输入和输出的array数据类型
        self.array_in_dtype = trt.nptype(self.engine.get_binding_dtype(0))
        self.array_out_dtype = trt.nptype(self.engine.get_binding_dtype(1))
        # 输入和输出的array数据shape（无batch）
        self.array_in_shape = self.engine.get_binding_shape(0)
        self.array_out_shape = self.engine.get_binding_shape(1)
        
        self.stream = cuda.Stream()
        self.h_input = cuda.pagelocked_empty(trt.volume(self.array_in_shape), dtype=self.array_in_dtype)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.array_out_shape), dtype=self.array_out_dtype)
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
    
    def run(self,data:np.ndarray)->np.ndarray:
        """
        data should be as batch.such as 128x3x32x32
        """
        with self.engine.create_execution_context() as context:
            return np.stack([self._compute(data[i],context) for i in range(data.shape[0])])                
    
    def compute(self,data:np.ndarray,context:trt.tensorrt.IExecutionContext)->np.ndarray:
        """
        data should single obj.such as 3x32x32
        """
        with self.engine.create_execution_context() as context:
            return self._compute(data)
        
    def _compute(self,data:np.ndarray,context:trt.tensorrt.IExecutionContext)->np.ndarray:
        assert (data.dtype == self.array_in_dtype)
        assert (data.shape == self.array_in_shape)
        
        self.h_input = data
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        # Run inference.
        context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return the host output. 
        return self.h_output.reshape(self.array_out_shape)
    
    def __str__(self):
        info = f"tensorrt engine \ninput  -> shape:{self.array_in_shape}\tdtype:{self.array_in_dtype}"+\
        f"\noutput -> shape:{self.array_out_shape}\tdtype:{self.array_out_dtype}"
        return info

    def __repr__(self):
        return self.__str__()
    
if __name__ == '__main__':
    
    image = np.random.rand(8,3,32,32)
    image = np.array(image,dtype=np.float16,order='C')
    print(image.shape)
    engine = Engine("resnet.trt")
    print(engine)
    result = engine.run(image)
    print(result.shape)
