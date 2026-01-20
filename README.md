# Video Markme - Player Tracking Application

תוכנה מקומית למעקב שחקניות בסרטונים והוספת סימונים ויזואליים.

## תכונות

- ✅ טעינת סרטונים עד 60 שניות (4K, 60fps)
- ✅ סימון ידני של שחקניות בפריים הראשון
- ✅ מעקב אוטומטי אחר כל שחקנית לאורך הסרטון
- ✅ 3 סגנונות סימון: חץ, עיגול זוהר, מלבן
- ✅ תמיכה במספר שחקניות במקביל
- ✅ ייצוא וידאו באיכות מקורית עם סאונד

## דרישות מערכת

- Python 3.10 או גבוה יותר
- FFmpeg מותקן וזמין ב-PATH
- Windows / Linux / macOS

## התקנה

**חשוב**: קרא את `INSTALL.md` להוראות מפורטות!

1. הסר גרסאות ישנות של OpenCV:
```bash
pip uninstall opencv-python opencv-python-headless -y
```

2. התקן את התלויות:
```bash
pip install -r requirements.txt
```

3. ודא ש-FFmpeg מותקן:
   - Windows: הורד מ-https://ffmpeg.org/download.html והוסף ל-PATH
   - Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian)
   - macOS: `brew install ffmpeg`

## שימוש

1. הפעל את התוכנה:
```bash
python app.py
```

2. לחץ על "Load Video" ובחר קובץ וידאו (mp4, mov, mkv, webm)

3. לחץ על "Add Player Marker" וצייר מלבן סביב השחקנית בפריים הראשון

4. בחר שם וסגנון סימון עבור השחקנית

5. חזור על שלבים 3-4 עבור שחקניות נוספות

6. לחץ על "Start Tracking" כדי להתחיל את תהליך המעקב

7. לאחר סיום המעקב, לחץ על "Export Video" כדי לייצא את הסרטון עם הסימונים

## מבנה הפרויקט

```
Video Markme/
├── app.py                 # נקודת כניסה ראשית
├── requirements.txt      # תלויות Python
├── PRD.md                # מסמך דרישות מוצר
├── README.md             # מדריך זה
└── src/
    ├── ui/               # רכיבי ממשק משתמש
    │   ├── main_window.py
    │   ├── video_canvas.py
    │   └── player_selector.py
    ├── tracking/         # מנוע מעקב
    │   ├── tracker_manager.py
    │   └── player_tracker.py
    └── render/           # רינדור וייצוא
        ├── overlay_renderer.py
        └── video_exporter.py
```

## הערות טכניות

- המעקב מבוסס על OpenCV (CSRT tracker)
- הייצוא משתמש ב-FFmpeg לשמירת איכות מקורית
- הסאונד מועבר ללא שינוי (passthrough)
- כל הפריימים מעובדים frame-by-frame למעקב מדויק

## פתרון בעיות

**שגיאת "TrackerCSRT not found" או "module 'cv2' has no attribute 'TrackerCSRT'":**
- זה אומר שצריך להתקין `opencv-contrib-python` במקום `opencv-python`
- הרץ:
```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python>=4.8.0
```

**FFmpeg לא נמצא:**
- ודא ש-FFmpeg מותקן וזמין ב-PATH
- נסה להריץ `ffmpeg -version` בטרמינל

**הסרטון לא נטען:**
- ודא שהפורמט נתמך (mp4, mov, mkv, webm)
- ודא שהסרטון לא עולה על 60 שניות
- ודא שהרזולוציה לא עולה על 4K

**מיקום תיבת הסימון לא מדויק:**
- ודא שחלון התוכנה לא מוקטן מדי
- נסה לשנות את גודל החלון ולנסות שוב

**מעקב לא מדויק:**
- ודא שהמלבן שציירת מכסה את השחקנית במלואה
- נסה להשתמש ב-CSRT tracker (ברירת מחדל)
- ודא שהתאורה בסרטון טובה

## רישיון

תוכנה זו נוצרה לשימוש מקומי בלבד.

# MarkMe2
