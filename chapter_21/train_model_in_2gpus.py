import torch
import torch.nn as nn
from accelerate import utils
from accelerate import Accelerator

# start a accelerate instance
utils.write_basic_config()
accelerator = Accelerator()
device = accelerator.device

def main():
    # define the model
    class MyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(len(w_list)))
        
        def forward(self, x:torch.Tensor):
            return self.w @ x

    # load training data
    import pickle
    with open("train_data.pkl",'rb') as f:
        loaded_object = pickle.load(f)
    w_list = loaded_object['w_list']
    x_list = loaded_object['input']
    y_list = loaded_object['output']

    # convert data to torch tensor
    x_input     = torch.tensor(x_list, dtype=torch.float32).to(device)
    y_output    = torch.tensor(y_list, dtype=torch.float32).to(device)

    # initialize model, loss function, and optimizer
    model       = MyLinear().to(device)
    loss_fn     = nn.MSELoss()
    optimizer   = torch.optim.SGD(model.parameters(), lr = 0.00001)

    # wrap model and optimizer using accelerate
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        for i, x in enumerate(x_input):
            # forward
            y_pred = model(x)

            # calculate loss
            loss = loss_fn(y_pred,y_output[i])

            # zero out the cached parameter.
            optimizer.zero_grad()

            # backward
            #loss.backward()
            accelerator.backward(loss)

            # update paramters
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    # take a look at the model weights after trainning
    model = accelerator.unwrap_model(model)
    print(model.w)

if __name__ == "__main__":
    main()