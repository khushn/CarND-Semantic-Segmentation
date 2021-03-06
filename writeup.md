### Apendix

All the main commands used


1. conda create --name sem_seg

2. Installing tensorflow for Python 3.6 and Only CPU version 
 pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
 NOTE: For GUP install the corresponding GPU enabled version

3. Installing NumPy
pip install --ignore-installed --upgrade https://pypi.python.org/packages/57/a7/e3e6bd9d595125e1abbe162e323fd2d06f6f6683185294b79cd2cdb190d5/numpy-1.13.3-cp36-cp36m-manylinux1_x86_64.whl#md5=bcbfbd9d0dbe026fd59a7756e190cdfa
Note: We need to select the right download for Python version, and Linux x86 architecture

4. INstalling SciPy
pip install --ignore-installed --upgrade https://pypi.python.org/packages/d8/5e/caa01ba7be11600b6a9d39265440d7b3be3d69206da887c42bef049521f2/scipy-1.0.0-cp36-cp36m-manylinux1_x86_64.whl#md5=9f77e8710fcab99ae4fed09d5fe56605
Note: For some reason, it installed NumPy again. 

5. Getting the data set
   * cd data/
   * wget <url provided by site http://www.cvlibs.net/download.php?file=data_road.zip>

6. Download pre-trained VGG model
	* comment out all the test.<calls> below all the functions in main.py, which are needed once the function is coded. 
	* And then run 'python main.py' (This will download the pretrained VGG model, when run for the first time)


7. GPU used
	g3.4xlarge instances which are NVIDIA Tesla M60 GPUs


### Building the AMI after each run
We need to do this, as no attaching any EBS (due to cost)

1)
aws ec2 create-image --instance-id i-09f2c138071f13788 --name "udacity-deep-learning-v3" --description "Run on g3.4xlarge" --no-reboot

NOTE: If the above command may not work, due to environment pointing to other zones, use the console to create the AMI, by giving name and description

2) Connecting to the instance

ssh -i udacity-deep-learning.pem ubuntu@<intance address>

### References

1. Nice advise on how to convolve skip layers to the same shape as the layer to which they are added: 
https://discussions.udacity.com/t/what-is-the-output-layer-of-the-pre-trained-vgg16-to-be-fed-to-layers-project/327033/24

2. Inputs used for data augmentation (For adding shade to images)
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

3. Description of AWS g3 GPUs
https://aws.amazon.com/about-aws/whats-new/2017/07/introducing-amazon-ec2-g3-instances-the-next-generation-of-gpu-powered-instances-for-graphics-intensive-applications/
