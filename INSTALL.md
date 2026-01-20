# הוראות התקנה - Video Markme

## שלב 1: התקן את Python
ודא שיש לך Python 3.10 או גבוה יותר מותקן.

## שלב 2: הסר גרסאות ישנות של OpenCV (חשוב!)
```bash
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
```

## שלב 3: התקן את התלויות
```bash
pip install -r requirements.txt
```

**חשוב**: אם השלב הזה נכשל, נסה:
```bash
pip install PyQt6>=6.6.0
pip install opencv-contrib-python>=4.8.0
pip install numpy>=1.24.0
pip install ffmpeg-python>=0.2.0
```

## שלב 4: התקן FFmpeg (חובה לשמירת אודיו!)
**חשוב מאוד**: FFmpeg נדרש כדי לשמור את הסאונד המקורי בקבצי הוידאו המיוצאים!

### macOS (המערכת שלך):
```bash
chmod +x scripts/install_ffmpeg.sh
./scripts/install_ffmpeg.sh
```

או באופן ידני:
```bash
brew install ffmpeg
```

### Windows:
1. הורד מ-https://ffmpeg.org/download.html
2. חלץ את הקבצים
3. הוסף את התיקייה `bin` ל-PATH

### Linux:
```bash
sudo apt-get install ffmpeg
```

### בדיקה שהכל עובד:
```bash
ffmpeg -version
```

אם אתה רואה את גרסת FFmpeg - הכל בסדר! ✅
אם אתה רואה "command not found" - חזור על שלב 4.

## שלב 5: הפעל את התוכנה
לחץ פעמיים על `run.bat` (Windows) או הרץ:
```bash
python app.py
```

## פתרון בעיות

### שגיאת "TrackerCSRT not found"
זה אומר ש-opencv-contrib-python לא מותקן. הרץ:
```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python>=4.8.0
```

### שגיאת "FFmpeg not found"
התקן FFmpeg ווודא שהוא ב-PATH.

### הווידאו לא נטען
ודא שהפורמט נתמך (mp4, mov, mkv, webm) ושהווידאו לא עולה על 60 שניות.







