from setuptools import setup, find_packages  
  
setup(  
    name = "run_tensorrt",  
    version = "1.0",  
    keywords = ("xavier", "jetson","tensortrt"),  
    description = "python3 tensorrt tools",  
    long_description = "input a tensorrt engine file.then use to inference.",  
    license = "MIT Licence",  
  
    url = "http://www.dnnmind.com",  
    author = "yumohc@gmail.com",  
    author_email = "yumohc@gmail.com",  
  
    packages = ['run_tensorrt'],  
    include_package_data = True,  
    platforms = "any",  
    install_requires = [],  
)