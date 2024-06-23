import torch
import torch.nn.functional
from torchvision import transforms

class Predictor():
    def __init__(self, model):
        self.model = model
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.n_classes = len(self.classes)
        self.img_size = 28
        
    #이미지를 예측하는 함수, 각 클래스 이름과 예측확룰을 튜플로 묶은 후 이들의 리스트를 만들어서 반환
    def predict(self, img):
        img = img.convert("L") #흑백으로
        img = img.resize((self.img_size, self.img_size)) #사이즈 변환
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0), (1.0))])
        img = transform(img) #이미지 가공
        x = img.reshape(1, 1, self.img_size, self.img_size) #배치가 1인 데이터로 만듦
        self.model.eval() #평가 모드
        y = self.model(x) #예측
        
        y_prob = torch.nn.functional.softmax(torch.squeeze(y)) #확률로 나타낸다.
        sorted_prob, sorted_indices = torch.sort(y_prob, descending = True)
        
        return [(self.classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]


