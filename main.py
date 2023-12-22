import os
import argparse
import cv2
import numpy as np
import json
import onnxruntime as ort

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        words = []
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                words.append(ix_to_word[str(int(ix))])
            else:
                break
        
        txt = ' '.join(words)
        out.append(txt.replace('@@ ', ''))

    return out

class Image_Caption():
    def __init__(self, encoder_modelpath, decoder_modelpath, vocpath):
        self.net = cv2.dnn.readNet(encoder_modelpath)
        self.input_height, self.input_width = 640, 640
        
        self.mean = np.array([0.485, 0.456, 0.406],
                             dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225],
                            dtype=np.float32).reshape((1, 1, 3))
        
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(decoder_modelpath, so)
        with open(vocpath) as f:
            self.vocab = json.loads(f.read())

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height),
                         interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32)/255.0 - self.mean) / self.std
        return img

    def detect(self, srcimg):
        img = self.preprocess(srcimg)
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        res = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        feat =  np.expand_dims(res.squeeze(axis=-1), axis=0)

        output = self.session.run(None, {self.session.get_inputs()[0].name: feat})
        # post processes
        seq, _ = output
        sents = decode_sequence(self.vocab, seq)
        return sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='testimgs/apple-490485_1920.jpg', help="image path")
    args = parser.parse_args()

    mynet = Image_Caption("weights/encoder.onnx", "weights/decoder_fc_rl.onnx", "weights/vocab.json")

    srcimg = cv2.imread(args.imgpath)
    sents = mynet.detect(srcimg)[0]
    print(sents)
    drawimg = srcimg.copy()
    cv2.putText(drawimg, sents, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite('result.jpg', drawimg)
    winName = 'Deep learning image caption in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()