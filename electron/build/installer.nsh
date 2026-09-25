; Custom NSIS hooks for WSP CameraTrap.
;
; WSP CameraTrap installs per user (no UAC, into %LOCALAPPDATA%\Programs),
; like the AddaxAI app it is built on. It shares nothing with an AddaxAI
; install on the same machine: its own registry key, its own data folder,
; and none of AddaxAI's Timelapse launcher shims (those belong to AddaxAI).
;
; Two responsibilities:
;
; 1. Publish where the exe is, under HKCU\Software\WSP-CameraTrap, so
;    scripts and support can find the install without guessing.
;
; 2. On uninstall, offer to delete the user's data folder
;    (%USERPROFILE%\WSP-CameraTrap). Never touches %USERPROFILE%\AddaxAI.

!define WSP_KEY "Software\WSP-CameraTrap"
!define WSP_DATA_DIR "$PROFILE\WSP-CameraTrap"

!macro customInstall
  WriteRegStr SHELL_CONTEXT "${WSP_KEY}" "ExePath" "$INSTDIR\${APP_EXECUTABLE_FILENAME}"
  WriteRegStr SHELL_CONTEXT "${WSP_KEY}" "InstallDir" "$INSTDIR"
  WriteRegStr SHELL_CONTEXT "${WSP_KEY}" "Version" "${VERSION}"
!macroend

!macro customUnInstall
  DeleteRegKey SHELL_CONTEXT "${WSP_KEY}"

  ; Skip the user-data prompt during silent uninstalls (scripted removal).
  IfSilent skip_userdata_removal
  IfFileExists "${WSP_DATA_DIR}\*.*" 0 skip_userdata_removal
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 \
    "Also delete your WSP CameraTrap user data?$\r$\n$\r$\n${WSP_DATA_DIR}$\r$\n$\r$\nThis includes your projects database, models, environments, and logs. Removing it cannot be undone.$\r$\n$\r$\nYour original images and videos are never touched.$\r$\n$\r$\nChoose No to keep your data so a future reinstall picks up where you left off." \
    /SD IDNO IDNO skip_userdata_removal
  RMDir /r "${WSP_DATA_DIR}"
  skip_userdata_removal:
!macroend
