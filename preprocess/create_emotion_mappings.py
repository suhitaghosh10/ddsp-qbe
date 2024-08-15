import os

import shutil
import librosa

# root_folder = '/project/sghosh/dataset/ESD'
# audio_folder = '/data/share/speech/Emotional Speech Dataset (ESD)'
# folders = ['0013','0014','0015','0016','0017','0018','0019','0020']

HAPPY = "happy"
SURPRISE = "surprise"
SAD = "sad"
NEU = "neutral"
ANGRY = "angry"

wavlm_root_path = '/scratch/sghosh/datasets/DDSP/libri_esd/train/wavlm'
audio_root_path = '/data/share/speech/Emotional Speech Dataset (ESD)'
out_path = '/project/sghosh/dataset/ESD'

folders = ['0013','0014','0015','0016','0017','0018','0019','0020']
emotions = ['Happy', 'Sad', 'Surprise', 'Neutral', 'Angry']

#copy wavlm and audio files to a common location
for folder in folders:
    for e in emotions:
        w_epath = os.path.join(wavlm_root_path, folder, e)
        a_epath = os.path.join(audio_root_path, folder, e)
        for sbf in ['train', 'test', 'evaluation']:
            wavlm_files = os.listdir(os.path.join(w_epath, sbf))
            audio_files = os.listdir(os.path.join(a_epath, sbf))
            for f in wavlm_files:
                shutil.copy(src=os.path.join(w_epath, sbf, f), dst=os.path.join(out_path, f))
                f = f.replace('.pt', '.wav')
                shutil.copy(src=os.path.join(a_epath, sbf, f), dst=os.path.join(out_path, f))

# map the files having different emotions but same content and speaker
emotions = []
data = []
content = []
refined_lines = []

for folder in folders:
    with open(os.path.join(audio_root_path, folder, folder+'.txt'), 'r', encoding = "ISO-8859-1") as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n', '')

            if line.startswith(folder):
                arr = line.split('\t')
                if len(arr)>2:
                    txt = arr[1]
                    wav = arr[0]
                else:
                    wav, txt = arr[0].split(' ',1)
                y, sr = librosa.load(os.path.join(out_path, wav + '.wav'), sr=16000)
                dur = librosa.get_duration(y=y, sr=sr)
                content.append(folder+'_'+txt)
                refined_lines.append(line+'\t'+folder)

emo_dict = {}
for i in content:
    emo_dict[i] = {HAPPY: None, SAD: None, SURPRISE: None, NEU: None, ANGRY: None, 'content':i}

for line in refined_lines:
    try:
        file_num, key, emotion, spkr = line.split('\t')
        emo_dict[spkr + '_'+ key][emotion.lower()] = file_num
    except:
        print(line.replace('\t', ';'))
        file_num_key, emotion, spkr = line.split('\t')
        file_num, key = file_num_key.split(' ',1)
        emo_dict[spkr + '_' + key][emotion.lower()] = file_num

ctr = 1
rm_keys = []
with open(os.path.join('/project/sghosh/code/ddsp-qbe/resources/emo_mapping.csv'), 'w', encoding = "ISO-8859-1") as file:
    file.write('id,'+HAPPY+','+SAD+','+SURPRISE+','+NEU+','+ANGRY+'\n')

    for key in emo_dict.keys():
        try:
            file.write(str(ctr)+ ','+
                       emo_dict[key][HAPPY]+ ','+
                       emo_dict[key][SAD]+ ','+
                       emo_dict[key][SURPRISE]+ ','+
                       emo_dict[key][NEU]+ ','+
                       emo_dict[key][ANGRY]+'\n')
            ctr +=1
        except:
            # ignore if having missing files
            print(print(emo_dict[key]))
            rm_keys.append(key)

for key in rm_keys:
    del emo_dict[key]

print(emo_dict)
import pickle
with open('/project/sghosh/code/ddsp-qbe/resources/emo_mapping.pkl', 'wb') as f:
    pickle.dump(emo_dict, f)

print(len(emo_dict.keys()))