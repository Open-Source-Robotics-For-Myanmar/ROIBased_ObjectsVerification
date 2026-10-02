# Manual ROI Tracker + Kalman Filter

ဒီ prototype က OpenCV ရဲ့ visual tracker ကိုသာ သုံးပါတယ်။ Object detection သို့မဟုတ် AI model မပါပါ။ စတင်ချိန်မှာ target ကို ကိုယ်တိုင် ROI ဆွဲရွေးပြီး tracker က နောက် frame တွေမှာ လိုက်ရှာပါတယ်။

## လိုအပ်ချက်

ZIP ကို folder တစ်ခုထဲ extract လုပ်ပါ။ Repository ရဲ့ root folder မှာ `.venv` ရှိပြီးသားဆိုရင် အဲဒီ root မှာ environment ကို activate လုပ်ပြီး extract လုပ်ထားတဲ့ folder ထဲဝင်ပါ:

```bash
source .venv/bin/activate
```

လိုအပ်တဲ့ package တွေ မရှိသေးရင် install လုပ်ပါ:

```bash
python -m pip install -r requirements.txt
```

`opencv-contrib-python` လိုအပ်ပါတယ်။ OpenCV build ထဲမှာ tracker API မပါရင် `CSRT`/`KCF`/`MOSSE` ဖန်တီးရာမှာ error ပြပါလိမ့်မယ်။

## စမ်းသပ်ရန်

Extract လုပ်ထားတဲ့ folder ထဲကနေ run ပါ:

```bash
python main.py
```

Default အနေနဲ့ camera index `0` ကိုဖွင့်ပါတယ်။ အခြား camera ကိုရွေးရန်:

```bash
python main.py --camera 1
```

Video ဖိုင်နဲ့ စမ်းရန်:

```bash
python main.py --video /path/to/video.mp4
```

ဖိုင်ဆုံးရင် အစကနေပြန်ဖွင့်ရန် `--loop` ထည့်ပါ:

```bash
python main.py --video /path/to/video.mp4 --loop
```

Tracker ရွေးချယ်စရာက `CSRT` (default), `KCF`, `MOSSE` ဖြစ်ပါတယ်။ ဥပမာ:

```bash
python main.py --video /path/to/video.mp4 --tracker KCF
```

## အသုံးပြုပုံ

- Target ပေါ်မှာ mouse ဘယ်ဘက်ခလုတ်ဖိပြီး box ဆွဲကာ လွှတ်ပါ။ ပထမဆုံးအကြိမ်လည်း ဒီနည်းနဲ့ရွေးပြီး၊ tracking နေစဉ်မှာလည်း ထပ်ဆွဲကာ target ပြန်ရွေးနိုင်ပါတယ်။
- `p`: pause/resume လုပ်ပါ။
- `q`: ပိတ်ပါ။
- OpenCV window ရဲ့ ညာဘက်အပေါ်မှာ processed-frame FPS ကို ပြပါတယ်။ Pause နေချိန်မှာ နောက်ဆုံးတန်ဖိုးကို ဆက်ပြထားပါတယ်။

အစိမ်းရောင် corner box နဲ့ အလယ် cross က tracker လိုက်နေတဲ့ target ကိုပြပါတယ်။ Tracker target ပျောက်သွားရင် box အနီရောင်ပြောင်းပြီး mouse နဲ့ ထပ်ဆွဲရွေးနိုင်ပါတယ်။

## Kalman နှင့် ကန့်သတ်ချက်

Kalman filter က target center `(cx, cy)` ရဲ့ position နဲ့ velocity ကို ခန့်မှန်းပြီး tracker measurement ရှိတဲ့အခါ position ကို ပြင်ဆင်ပေးပါတယ်။ Tracker target ပျောက်သွားရင် measurement အတုမထည့်ဘဲ prediction ကိုသာ ဆက်ပြပါတယ်။ ပြန်မတွေ့နိုင်ရင် mouse နဲ့ ROI ကို ကိုယ်တိုင်ပြန်ရွေးပါ။