01. same with the lecture
02. remove similar illness, goes worse
03. remove mental problems, more epochs, larger train loss, less valid loss.
04. remove state, best for less train&valid loss and less epoch, rain Loss: 1.1178, Valid Loss: 1.2477

based on 04:
05, batch size = 128, Epoch [1682/3000]: Train Loss: 1.0909, Valid Loss: 1.2297
06, batch size = 128, lr = 1e-6, Epoch [3000/3000]: Train Loss: 1.2012, Valid Loss: 1.3321, goes worse
07, batch size = 128, validation ratio=0.25, Epoch [3000/3000]: Train Loss: 1.2012, Valid Loss: 1.3321
08, batch size = 128, va_ratio = 0.15, Epoch [1628/3000]: Train Loss: 1.0846, Valid Loss: 1.2999
09, batch size = 64, Epoch [1949/3000]: Train Loss: 1.0850, Valid Loss: 1.2283, model info was rewrite by model10

10, batch size = 128, seed = 3000, Epoch [1553/6000]: Train Loss: 1.1135, Valid Loss: 1.0531
11, batch size = 128, seed = 30000, lr = 1e-6, stop=800, all=10000, poch [5067/10000]: Train Loss: 1.1532, Valid Loss: 1.1795
12, batch size = 128, seed = 3000, lr = 5e-6, stop=800, all=10000, poch [5067/10000]: Train Loss: 1.1532, Valid Loss: 1.1795
13, batch size = 128, seed = 5201314, lr = 2e-5, stop=500, all=3000, Epoch [1782/3000]: Train Loss: 1.0883, Valid Loss: 1.3120

14, 32-16-1, Epoch [1470/3000]: Train Loss: 1.0815, Valid Loss: 1.2455
15, 32-16-8-1, Epoch [1454/3000]: Train Loss: 1.0910, Valid Loss: 1.2613
16, momentum = 0.45, Epoch [2874/3000]: Train Loss: 1.1042, Valid Loss: 1.2487
17, Adapgrad, lr=0.01, Epoch [1949/3000]: Train Loss: 1.0967, Valid Loss: 1.3439  ## ok
18, momentum = 0.9, data to 2022, Epoch [1682/3000]: Train Loss: 1.0909, Valid Loss: 1.2297
19, momentun=0.8, data to 2022, lr=9e-6, Epoch [2417/3000]: Train Loss: 1.0959, Valid Loss: 1.3897
20, leakyrelu, dropout, ...

# 5, 13, 15, 18, 19,

Show loss online with tensorboard:
tensorboard --logdir "./hw1/runs/recall"