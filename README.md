⚠ Under construction
---
But the current demo version is stable.
You need Python 3.10+ (but not 3.12 as it does not support tensorflow)
TensorFlow 2.12 is the last version officially supporting Python 3.11 on Windows.

Install dependencies:
`pip install -r requirements.txt`

Install spaCy model
`python -m spacy download en_core_web_sm`

Run in streamlit:
`python -m streamlit run app.py`

Common problems:
1. ```bash
   ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
   A dynamic link library (DLL) initialization routine failed.
    ```
| Cause                                      | Explanation                                                                                     |
|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Missing Visual C++ Redistributables** | TensorFlow needs specific low-level system DLLs (`MSVCP140.dll`, etc.) from Microsoft. Download the [latest version](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) of C++ and also VC++ from [here](https://aka.ms/vs/17/release/vc_redist.x64.exe)       |
| **Incompatible TensorFlow version**     | If you mix TensorFlow versions with Python versions it doesn't support, it breaks like this.    |
| **GPU-related issues** (if applicable)  | Installing `tensorflow-cpu`, sometimes the DLLs still call GPU-related imports. So install the lightweight cpu version  |
| **Windows blocks DLL loading**          | Sometimes SmartScreen/Antivirus blocks DLL initialization silently.                             |


