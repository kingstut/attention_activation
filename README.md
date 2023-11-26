## Attention with activation 
Zeroes the outputs of the heads whose sum of elements is less than zero (think ReLU for attention heads)
```python
y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

# head activation
if self.head_activation:  
    sum_y = torch.sum(y, dim=(-2, -1), keepdim=True)
    mask = sum_y < 0
    mask_expanded = mask.expand(-1, -1, T, C // self.n_head)
    y[mask_expanded] = 0

```