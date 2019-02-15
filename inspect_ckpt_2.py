# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("./save_restore_model/my_test", tensor_name='', all_tensors=True)
# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("./save_restore_model/my_test", tensor_name='v1', all_tensors=False)
# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("./save_restore_model/my_test", tensor_name='v2', all_tensors=False)