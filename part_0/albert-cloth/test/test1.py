# guess = [1, 0, 3, 2, 1, 0, 1, 3, 1, 2, 1, 0, 3, 2, 3, 2, 0, 1, 3, 1]
# ans = ["B", "A", "D", "C", "B", "A", "B", "D", "C", "C", "B", "A", "D", "C", "A", "C", "A", "D", "D", "B"]
# [1, 0, 2, 3, 2, 0, 2, 3, 1, 0, 0, 0, 2, 3, 1, 1, 0, 1, 2, 3, 
#  1, 0, 3, 2, 1, 0, 1, 3, 1, 2, 1, 0, 3, 2, 3, 2, 0, 1, 3, 1]
# for i in range(len(ans)):
#     if ans[i] == "A":
#         ans[i] = 0
#     elif ans[i] == "B":
#         ans[i] = 1
#     elif ans[i] == "C":
#         ans[i] = 2
#     elif ans[i] == "D":
#         ans[i] = 3

# print(ans)
# print(guess)
import torch
t = torch.load('guess1.pt')
torch.set_printoptions(threshold=5000)
print(t)