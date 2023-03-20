import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output

am_inference_dir = "./"#你的模型路径
voc_inference_dir = "./pwgan_aishell3_static_1.1.0"
wav_output_dir = "./inference_demo"
device = "gpu"

frontend = get_frontend(
    lang="mix",
    phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
    tones_dict=None
)

am_predictor = get_predictor(
    model_dir=am_inference_dir,
    model_file="fastspeech2_mix" + ".pdmodel",
    params_file="fastspeech2_mix" + ".pdiparams",
    device=device)

voc_predictor = get_predictor(
    model_dir=voc_inference_dir,
    model_file="pwgan_aishell3" + ".pdmodel",
    params_file="pwgan_aishell3" + ".pdiparams",
    device=device)

output_dir = Path(wav_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

fs = 24000

def generate_and_save():
    text = text_box.get("1.0", tk.END).strip()
    if text:
        am_output_data = get_am_output(
            input=text,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=True,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0, )
        wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialdir=os.getcwd(),
            initialfile=f"{text}.wav")
        if file_path:
            sf.write(file_path, wav, samplerate=fs)
            status_label.config(text=f"文件已保存到 {file_path}")
    else:
        status_label.config(text="请输入文本！")

root = tk.Tk()
root.title("语音合成工具 by XiaozhouTAT")

text_label = tk.Label(root, text="请输入要生成的文本：")
text_label.pack()
text_box = tk.Text(root, height=10, width=50)
text_box.pack()

button_frame = tk.Frame(root)
button_frame.pack()
generate_button = tk.Button(button_frame, text="生成并保存", command=generate_and_save)
generate_button.pack(side=tk.LEFT)
exit_button = tk.Button(button_frame, text="退出", command=root.destroy)
exit_button.pack(side=tk.RIGHT)

status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()
