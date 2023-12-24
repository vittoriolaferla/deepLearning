from torch.utils.data import DataLoader, Dataset

class QADataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions['input_ids']) 

    def __getitem__(self, idx):
        return {
            'input_ids': self.questions['input_ids'][idx],
            'labels': self.answers['input_ids'][idx],
        }

