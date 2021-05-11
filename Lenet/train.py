from model import Lenet
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

def main():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_sets = torchvision.datasets.CIFAR10(root="./data",train = True,
                                              download= False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_sets,batch_size = 36,
                                                     shuffle = True,num_workers = 0 )

    val_sets = torchvision.datasets.CIFAR10(root="./data",train=False,
                                            download=False,transform = transform)
    val_loader = torch.utils.data.DataLoader(val_sets,batch_size = 1000,
                                                   shuffle = False, num_workers = 0)
    val_data_iter = iter(val_loader)
    val_images,val_labels = val_data_iter.next()


    net = Lenet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = .001)

    for epoch in range(10):
        running_loss = 0.0
        for step,data in enumerate(train_loader):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_images)
                    predict_y = torch.max(outputs,dim = 1)[1]
                    accuracy =  torch.eq(predict_y,val_labels).sum().item()/val_labels.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print("Finishing Training")
    sava_path = "./Lenet.pth"
    torch.save(net.state_dict(),sava_path)


if __name__ == "__main__":
    main()