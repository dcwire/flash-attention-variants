from flash_attention.backends import Backend
from flash_attention.reference.naive import naive_attention

# The oracle. supports_backward=True comes for free from autograd once the forward exists.
BACKEND = Backend(name="torch_naive", attention=naive_attention, supports_backward=True)
