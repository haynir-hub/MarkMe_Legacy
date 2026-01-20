# 📝 Video Markme - Change Log

## 🎯 תיקוני באגים - 22/11/2025 (v4.1)

### 🔧 תיקונים:

#### 1️⃣ תיקון TypeError ב-_update_buttons
**בעיה:** `TypeError: setEnabled(self, a0: bool): argument 1 has unexpected type 'NoneType'`
**פתרון:** שימוש ב-`bool()` wrapper ל-`has_players`
**קובץ:** `src/ui/main_window.py`

#### 2️⃣ תיקון ניווט פריימים במצב Batch
**בעיה:** כפתורי Previous/Next Frame לא עבדו, Frame counter היה 0/0
**פתרון:** 
- עדכון כל פונקציות הניווט להשתמש ב-`project_manager.get_current_project()`
- תיקון 6 פונקציות: `_prev_frame`, `_next_frame`, `_show_frame`, `_update_frame_info`, `_update_frame_navigation_buttons`, `_on_video_selected`
**קובץ:** `src/ui/main_window.py`

#### 3️⃣ תיקון מיקום העיגול בסימון ראשוני
**בעיה:** העיגול לא היה ממורכז סביב השחקנית כבר בסימון הראשוני (לפני tracking)
**פתרון:** 
- שינוי נוסחת חישוב `center_y` ל-`y + h + int(radius_y * 0.3)`
- העיגול עכשיו ממורכז מדויק סביב הרגליים מההתחלה
**קובץ:** `src/render/overlay_renderer.py`

### 📁 קבצים שהשתנו:
- ✅ `src/ui/main_window.py` - תיקוני ניווט וכפתורים
- ✅ `src/render/overlay_renderer.py` - תיקון מיקום עיגול
- ✅ `HOTFIX_BATCH.txt` - תיעוד תיקון TypeError
- ✅ `HOTFIX_NAVIGATION.txt` - תיעוד תיקון ניווט
- ✅ `FIX_CIRCLE_INITIAL_CENTER.txt` - תיעוד תיקון עיגול

### 🎯 תוצאה:
✅ כל הכפתורים עובדים תקין
✅ ניווט פריימים עובד מושלם
✅ העיגול ממורכז נכון כבר מההתחלה

---

## 🎬 Batch Processing - עיבוד מספר סרטונים! - 21/11/2025 (v4)

### 🚀 תכונה מרכזית חדשה: Batch Processing

**מה זה?**
עכשיו אפשר לטעון מספר סרטונים, לסמן שחקניות בכל אחד, ולייצא את כולם בלחיצה אחת!

### ✨ מה נוסף:

#### 1️⃣ מבנה נתונים חדש:
✅ **VideoProject** (`src/tracking/video_project.py`):
   - מייצג פרויקט סרטון בודד
   - מטא-דאטה, שחקנים, תוצאות tracking
   - סטטוסים: pending, marked, tracked, exported, failed, skipped

✅ **ProjectManager** (`src/tracking/project_manager.py`):
   - מנהל מספר פרויקטים במקביל
   - ניהול הפרויקט הנוכחי
   - זיהוי פרויקטים מוכנים לייצוא

✅ **BatchExportThread** (`src/render/batch_exporter.py`):
   - עובר על כל הסרטונים ברצף
   - מבצע tracking + export לכל סרטון
   - מדלג אוטומטית על סרטונים ללא שחקנים
   - מדווח על הצלחות וכישלונות

#### 2️⃣ ממשק משתמש חדש:

📹 **רשימת סרטונים (Video List):**
   - מציג את כל הסרטונים שנטענו
   - סטטוס ויזואלי עם סמלים:
     * ⏸️ Pending - נטען, אין שחקנים
     * ✏️ Marked - יש שחקנים, מוכן
     * 🔄 Tracking - בתהליך tracking
     * ✅ Tracked - tracking הושלם
     * 📤 Exporting - בתהליך ייצוא
     * 🎬 Exported - ייצוא הושלם
     * ❌ Failed - נכשל
     * ⏭️ Skipped - דולג (אין שחקנים)
   - כפתורים: Add Videos, Remove

👥 **רשימת שחקנים לסרטון נוכחי:**
   - כל סרטון יכול לקבל שחקנים שונים
   - הוספה והסרה בקלות

🎬 **כפתורי ייצוא חדשים:**
   - "📤 Export All Videos" - מייצא את כולם ברצף
   - "📤 Export Current Video" - מייצא רק את הנוכחי

#### 3️⃣ תהליך עבודה חדש:

```
שלב 1: טען סרטונים
➕ Add Videos → בחר מספר סרטונים

שלב 2: סמן שחקנים בכל סרטון
לחץ על סרטון → Add Marker → צייר bbox → בחר סגנון

שלב 3: ייצא הכל בלחיצה אחת!
📤 Export All Videos → בחר תיקייה → המערכת תעשה הכל

המערכת:
✅ תבצע tracking אוטומטית לכל סרטון
✅ תייצא עם הסימונים
✅ תדלג על סרטונים ללא שחקנים
✅ תדווח על כישלונות
```

#### 4️⃣ תכונות חכמות:

✅ **דילוג אוטומטי:**
   - סרטונים ללא שחקנים נדלגים
   - הודעה ברורה: "Skipped: No players marked"

✅ **טיפול בכישלונות:**
   - סרטון שנכשל מקבל סטטוס ❌ Failed
   - הודעת שגיאה מפורטת
   - המערכת ממשיכה לסרטון הבא

✅ **דו"ח סיכום:**
   - "✅ Successful: X/Y videos"
   - "❌ Failed: X/Y videos"

✅ **UI ריאקטיבי:**
   - כפתורים מתאימים מופעלים/מושבתים
   - Progress bar לכל סרטון
   - עדכוני סטטוס בזמן אמת

### 📁 קבצים שהשתנו/נוספו:
- ✅ `src/tracking/video_project.py` (חדש!)
- ✅ `src/tracking/project_manager.py` (חדש!)
- ✅ `src/render/batch_exporter.py` (חדש!)
- ✅ `src/ui/main_window.py` (שינויים משמעותיים)
- ✅ `src/ui/main_window_old_backup.py` (גיבוי)
- ✅ `BATCH_PROCESSING_GUIDE.txt` (מדריך מפורט!)

### 🎯 יתרונות:
✅ עיבוד מספר סרטונים בבת אחת
✅ חיסכון עצום בזמן
✅ מעקב אחר כל הסרטונים במקום אחד
✅ דילוג אוטומטי על בעיות
✅ דו"ח מפורט בסיום

### 💡 דוגמה:
```
טען 10 סרטונים של משחק → סמן שחקניות בכל אחד → 
לחץ Export All → קבל 10 סרטונים מעוצבים עם tracking!

במקום לעשות את זה 10 פעמים ידנית 🎉
```

---

## 🎯 שיפור מיקום העיגול - השחקנית במרכז - 21/11/2025 (v3)

### תיקון מרכזי:

**בעיה:** העיגול לא היה ממורכז סביב השחקנית במהלך ה-tracking

**פתרון:**
1. ✅ הגדלת smoothing buffer מ-5 ל-**10 פריימים** - תנועה חלקה יותר
2. ✅ שיפור חישוב מרכז העיגול:
   - `center_y = y + h - (h * 0.02)` - 2% מעל התחתית במקום בדיוק בתחתית
   - זה מבטיח שהעיגול ממוקם מדויק יותר סביב הרגליים
3. ✅ הגדלת רדיוסי העיגול מ-60%/20% ל-**65%/22%** - עיגול קצת יותר גדול
4. ✅ הוספת debug logging עם 🎯 סימן לבדיקה

**קבצים שהשתנו:**
- `src/tracking/player_tracker.py` - buffer_size=10
- `src/render/overlay_renderer.py` - מיקום מדויק + debug

**תוצאה:** העיגול צריך להיות ממורכז מדויק על הרגליים של השחקנית 🎯

---

## 🎨 שדרוג מקצועי - סגנונות סימון + תיקון ייצוא - 21/11/2025 (v2)

### ✨ תיקונים מרכזיים:

#### 1️⃣ תיקון Export Error (הסרטון היה תמונה דוממת)
**בעיה:**
- FFmpeg לא מצא קבצי פריימים: `Could find no file with path`
- הסרטון המיוצא היה תמונה סטטית
- המערכת קפאה במהלך הייצוא

**פתרון:**
- שיניתי את `video_exporter.py` להשתמש ב-`tracker_manager.get_frame()` במקום `cap.read()`
- זה מבטיח קריאה מדויקת עם 3 אסטרטגיות הגיבוי
- כל הפריימים נכתבים כעת בהצלחה
- הוספתי logging כל 50 פריימים

#### 2️⃣ שדרוג העיגול ל-3D Floor Hoop ⭕
**בעיה:**
- העיגול נראה סטטי ושטוח
- לא נראה כמו חישוק על הפרקט

**פתרון:**
- שיניתי מעיגול (`cv2.circle`) לאליפסה (`cv2.ellipse`)
- רדיוס אופקי רחב (60% מרוחב), רדיוס אנכי דחוס (20% מרוחב)
- זה יוצר אפקט פרספקטיבה של חישוק מונח על הרצפה
- הוספתי קו פנימי בהיר ליצירת עומק
- **נראה בדיוק כמו בשידורי NBA/FIFA מקצועיים!** 🏀⚽

#### 3️⃣ שינוי הריבוע לכחול ללא מילוי 🔷
**בעיה:**
- הריבוע היה אדום עם מילוי שקוף

**פתרון:**
- צבע כחול בהיר (`(255, 100, 0)`)
- הסרתי את המילוי השקוף
- הוספתי פינות מודגשות (corner highlights) בצבע cyan
- אפקט glow חיצוני עדין
- נראה מודרני ומקצועי

#### 4️⃣ הוספת סגנונות סימון מקצועיים חדשים! ✨

**💡 Spotlight Effect:**
- אפקט זרקור אצטדיון
- תאורה עגולית עם גרדיאנט חלק
- מתאים להדגשת השחקנית המרכזית

**✨ Professional Glow Outline:**
- קווי מתאר זוהרים סביב השחקנית
- אפקט glow חיצוני עם שכבות מרובות
- קו פנימי בהיר לעומק
- מושלם לשחקניות כוכבות

**סה"כ כעת 5 סגנונות:**
1. 🔺 Arrow above head
2. ⭕ 3D Floor Hoop (כמו בטלוויזיה!)
3. 🔷 Blue Rectangle with corners
4. 💡 Spotlight effect
5. ✨ Professional Glow Outline

### 📁 קבצים שהשתנו:
- `src/render/video_exporter.py` - תיקון קריאת פריימים לייצוא
- `src/render/overlay_renderer.py` - 3D ellipse, ריבוע כחול, סגנונות חדשים
- `src/ui/player_selector.py` - UI עם 5 סגנונות + תיאורים
- `src/tracking/tracker_manager.py` - מיפוי צבעים לסגנונות החדשים

### 🎯 תוצאה:
✅ ייצוא וידאו עובד מושלם
✅ עיגול תלת-ממדי מקצועי
✅ ריבוע כחול נקי
✅ 5 אפשרויות סימון שונות
✅ נראה כמו בטלוויזיה! 📺

---

## 🎯 תיקון קריטי - ניווט פריימים - 21/11/2025 (v1)

### 🔧 בעיה: הפריימים לא השתנו בתצוגה
**תסמינים:**
- הכפתורים עבדו (המספרים השתנו)
- אבל התצוגה נשארה תקועה על הפריים הראשון
- השגיאה: `ERROR: Failed to read frame X` (frame 1, 2, 3...)

**גילוי הבעיה:**
- הפריים 0 נקרא בהצלחה
- כל הפריימים האחרים החזירו `ret=False`
- `video_cap.set(cv2.CAP_PROP_POS_FRAMES)` לא עבד (בעיית seeking)

**הפתרון:**
יצרתי 3 אסטרטגיות גיבוי לקריאת פריימים:
1. **Strategy 1** - Reset + Seek (הכי מהיר)
2. **Strategy 2** - VideoCapture חדש + Seek (אמצע)
3. **Strategy 3** - Sequential Read מהתחלה (הכי אמין, עובד תמיד)

המערכת תנסה כל אסטרטגיה עד שאחת מצליחה.

**קבצים שהשתנו:**
- `src/tracking/tracker_manager.py` - פונקציה `get_frame` עם 3 אסטרטגיות
- `src/ui/main_window.py` - ניקוי הדפסות

**הערה:** אם האסטרטגיה השלישית עובדת, הניווט יהיה איטי יותר. זה בגלל הקודק של הוידאו. המלצה: המר ל-H.264/MP4.

---

## תיקונים קודמים - 21/11/2025

### 🔧 בעיה 1: כפתורי הניווט לא עבדו
**תסמינים:**
- לחיצה על "Previous Frame" / "Next Frame" לא גרמה לשינוי
- הפריימים לא התקדמו

**התיקון:**
- הוספתי הדפסות debug ל-console כדי לעקוב אחרי הלחיצות
- וידאתי שהלוגיקה של `_prev_frame` ו-`_next_frame` עובדת תקין
- בדקתי ש-`total_frames` מוגדר נכון לאחר טעינת הוידאו

**קבצים שהשתנו:**
- `src/ui/main_window.py` - פונקציות `_prev_frame`, `_next_frame`

---

### 🔧 בעיה 2: המערכת קפאה בזמן ייצוא
**תסמינים:**
- בלחיצה על "Export Video" המערכת נעשתה "Not Responding"
- לא ניתן היה לראות את סרגל ההתקדמות מתקדם
- החלון היה מוקפא עד לסיום הייצוא

**התיקון:**
- יצרתי `ExportThread` חדש שרץ ברקע (בדומה ל-`TrackingThread`)
- העברתי את כל תהליך הייצוא ל-thread נפרד
- עכשיו הממשק נשאר מגיב וסרגל ההתקדמות מתעדכן בזמן אמת

**קבצים שהשתנו:**
- `src/ui/main_window.py` - הוספת `ExportThread`, `_on_export_progress`, `_on_export_finished`

---

### 🔧 בעיה 3: הוידאו המיוצא היה תמונה דוממת
**תסמינים:**
- הקובץ שיצא נראה כמו תמונה אחת
- הווידאו לא התנגן או הראה רק פריים אחד
- משך הסרטון היה 0 או 1 שניה

**התיקון:**
1. **תיקון קריאת הפריימים:**
   - הוספתי בדיקה שכל הפריימים נקראים ונשמרים
   - הוספתי ספירה של `frames_written` לוודא שכל הפריימים נכתבו

2. **תיקון פקודת FFmpeg:**
   - שיניתי `-r` ל-`-framerate` בקלט (משפיע על קצב קריאת הפריימים)
   - הוספתי `-r` גם בפלט כדי לקבוע את ה-FPS הסופי
   - שיניתי preset מ-`slow` ל-`fast` לתאימות טובה יותר
   - בשלב החיבור עם אודיו, שיניתי מ-re-encode ל-`copy` כדי לא לאבד איכות

3. **הוספת הדפסות מפורטות:**
   - "Processing X frames..."
   - "Processed X frames out of Y"
   - "Creating video from frames at 30 FPS..."
   - "Video created successfully"

**קבצים שהשתנו:**
- `src/render/video_exporter.py` - פונקציות `export_video`, `_export_with_ffmpeg`

---

## 🎯 איך לבדוק שהתיקונים עובדים

### בדיקת ניווט:
1. הפעל את התוכנה: `python app.py` או לחץ על `run.bat`
2. טען וידאו
3. לחץ על "Next Frame" מספר פעמים
4. בדוק ב-console (PowerShell) - אמור לראות:
   ```
   _next_frame called: current=0, total=150
   Moving to frame 1
   _next_frame called: current=1, total=150
   Moving to frame 2
   ```

### בדיקת ייצוא:
1. עשה tracking על שחקנית
2. לחץ על "Export Video"
3. המערכת לא צריכה להקפיא
4. תראה סרגל התקדמות מתקדם
5. אחרי סיום, נגן את הקובץ המיוצא - אמור להיות סרטון מלא ולא תמונה

### בדיקת הדפסות:
פתח את PowerShell ובדוק שרואה הדפסות כמו:
```
Processing 150 frames...
Processed 150 frames out of 150
Creating video from frames at 30.0 FPS...
FFmpeg command: ffmpeg -y -framerate 30.0 -i ...
Video created successfully
Combining video with audio...
Video exported with audio successfully
```

---

## 📦 קבצים חדשים שנוספו
- `TEST_INSTRUCTIONS.txt` - הוראות בדיקה מפורטות
- `CHANGELOG.md` - מסמך זה
- `INSTALLATION_STATUS.txt` - סטטוס התקנה

## 📚 קבצים קיימים שעודכנו
- `src/ui/main_window.py` - תיקוני ניווט וייצוא async
- `src/render/video_exporter.py` - תיקון ייצוא וידאו
- `requirements.txt` - הוספת opencv-contrib-python

---

## 💡 טיפים
- אם עדיין יש בעיות, שלח צילום מסך + copy/paste של כל התוכן מ-PowerShell
- ודא ש-FFmpeg מותקן: `ffmpeg -version`
- ודא ש-opencv-contrib-python מותקן: `pip list | findstr opencv`
- במקרה של בעיה, הרץ: `scripts/reinstall.bat`

