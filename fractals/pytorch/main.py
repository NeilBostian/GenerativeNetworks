from model_processor import ModelProcessor

if __name__ == '__main__':
    m = ModelProcessor('.data/model')
    if not m._loss_trained:
        m.train_loss()
    m.train_frames()