import torch
import torch.nn.functional as F

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            logits1 = model(img1)
            logits2 = model(img2)

            cos_sim = F.cosine_similarity(logits1, logits2)
            predicted = (cos_sim > 0.5).long()

            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    return accuracy
