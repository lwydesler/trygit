# 导入相关模块
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
import math

# 定义超参数
BATCH_SIZE = 128
LR = 0.01
NUM_EPOCHS = 10
SRC_VOCAB_SIZE = 10000 # 源语言词汇表大小
TGT_VOCAB_SIZE = 10000 # 目标语言词汇表大小
EMB_SIZE = 512 # 词嵌入维度
NHEAD = 8 # 多头注意力头数
FFN_HID_DIM = 2048 # 前馈网络隐藏层维度
NUM_ENCODER_LAYERS = 6 # 编码器层数
NUM_DECODER_LAYERS = 6 # 解码器层数

# 定义数据集和数据加载器
train_dataset, valid_dataset, test_dataset = torchtext.datasets.IWSLT2017(language_pair=('de', 'en'))
src_tokenizer = get_tokenizer('spacy', language='zh_core_news_sm')
tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
src_vocab = torchtext.vocab.build_vocab(train_dataset, key=lambda x: src_tokenizer(x[0]), max_size=SRC_VOCAB_SIZE)
tgt_vocab = torchtext.vocab.build_vocab(train_dataset, key=lambda x: tgt_tokenizer(x[1]), max_size=TGT_VOCAB_SIZE)
src_pad_idx = src_vocab['<pad>']
tgt_pad_idx = tgt_vocab['<pad>']
train_dataloader = torchtext.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: torchtext.data.batch.batch(batch, src_pad_idx, tgt_pad_idx))
valid_dataloader = torchtext.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: torchtext.data.batch.batch(batch, src_pad_idx, tgt_pad_idx))
test_dataloader = torchtext.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: torchtext.data.batch.batch(batch, src_pad_idx, tgt_pad_idx))

# 定义Transformer模型
model = nn.Transformer(d_model=EMB_SIZE, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                       num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=FFN_HID_DIM)
src_emb = nn.Embedding(SRC_VOCAB_SIZE, EMB_SIZE)
tgt_emb = nn.Embedding(TGT_VOCAB_SIZE, EMB_SIZE)
pos_enc = nn.PositionalEncoding(EMB_SIZE)
generator = nn.Linear(EMB_SIZE, TGT_VOCAB_SIZE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 定义训练和评估函数
def train(model, dataloader):
    model.train()
    total_loss = 0.0
    for src, tgt in dataloader:
        src = src_emb(src) * math.sqrt(EMB_SIZE)
        src = pos_enc(src)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        tgt_input = tgt_emb(tgt_input) * math.sqrt(EMB_SIZE)
        tgt_input = pos_enc(tgt_input)
        optimizer.zero_grad()
        output = model(src, tgt_input)
        output = generator(output)
        loss = criterion(output.view(-1, TGT_VOCAB_SIZE), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src_emb(src) * math.sqrt(EMB_SIZE)
            src = pos_enc(src)
            tgt_input = tgt[:-1,:]
            tgt_output = tgt
            tgt_input = tgt_emb(tgt_input) * math.sqrt(EMB_SIZE)
            tgt_input = pos_enc(tgt_input)
            output = model(src, tgt_input)
            output = generator(output)
            loss = criterion(output.view(-1, TGT_VOCAB_SIZE), tgt_output.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

    # 定义主程序


if __name__ == "__main__":
    # 训练模型
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_dataloader)
        valid_loss = evaluate(model, valid_dataloader)
        print(f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}")
    # 测试模型
    test_loss = evaluate(model, test_dataloader)
    print(f"Test loss: {test_loss:.4f}")