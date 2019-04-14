import os
import tensorrt as trt
import torch

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
        self.array_in_dtype = self.engine.get_binding_dtype(0)
        self.array_out_dtype = self.engine.get_binding_dtype(1)
        # 输入和输出的array数据shape（无batch）
        self.array_in_shape = self.engine.get_binding_shape(0)
        self.array_out_shape = self.engine.get_binding_shape(1)
        
        self.batch_size = self.engine.max_batch_size

        
    def run(self,data:torch.Tensor)->torch.Tensor:
        """
        data should be as batch.such as 128x3x32x32
        """
        data_out = torch.zeros_like(data,device=data.device,dtype=data.dtype)
        i = 0
        r = data.shape[0]
        assert(r % self.batch_size == 0)
        while i<r:
            self._compute(data[i].data_ptr(),data_out[i].data_ptr(),batch_size = self.batch_size)              
            i += self.batch_size
        return data_out
    
    def _compute(self,data_in_index:int,data_out_index:int,batch_size:int = 1):
        if not hasattr(self,'context'):
            self.context = self.engine.create_execution_context()
        self.context.execute(batch_size,[data_in_index,data_out_index])
    
    def __str__(self):
        info = f"tensorrt engine \ninput  -> shape:{self.array_in_shape}\tdtype:{self.array_in_dtype}"+\
        f"\noutput -> shape:{self.array_out_shape}\tdtype:{self.array_out_dtype}\n"+\
        f"batch size:{self.batch_size}"
        return info

    def __repr__(self):
        return self.__str__()
    
if __name__ == '__main__':
    from tqdm import tqdm
    image = torch.rand(32,3,32,32,device='cuda',dtype=torch.half)
    print(image.shape)
    engine = Engine("resnet.trt")
    print(engine)
    for i in tqdm(range(100)):
        result = engine.run(image)
