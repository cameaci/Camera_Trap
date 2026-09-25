/**
 * Headless bridge between the native Electron application menu and the
 * React app. The menu is built in the Electron main process; renderer-backed
 * items send a string command over the "menu:command" channel and this
 * component runs the matching action (navigation, one of the app dialogs, a
 * folder-open, the diagnostic export, or the species-name toggle).
 *
 * This is the successor to the old AppHamburger dropdown: same actions, same
 * dialogs, no button UI. It renders nothing except the dialogs it owns.
 * Mounted once at the app root (next to DownloadCompleteToasts in App.tsx),
 * inside the router and query provider so navigation and queries work.
 *
 * Menu commands only arrive in Electron, where window.electronAPI is defined.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { setupApi } from "../../api/setup";
import { backupApi } from "../../api/backup";
import { useLatestRelease } from "../../hooks/useLatestRelease";
import { formatVersion, parseVersion } from "../../lib/version";
import { exportDiagnosticReport } from "../../lib/diagnostic-export";
import { getSpeciesNameMode, setSpeciesNameMode } from "../../lib/species-name-mode";
import { ResetAppDialog } from "../diagnostics/ResetAppDialog";
import { BackupNowDialog } from "../diagnostics/BackupNowDialog";
import { RestoreBackupDialog } from "../diagnostics/RestoreBackupDialog";
import { CheckForUpdatesDialog } from "../diagnostics/CheckForUpdatesDialog";
import { ModelLibraryDialog } from "../diagnostics/ModelLibraryDialog";

type DialogId = "reset" | "updates" | "backup" | "restore" | "model-library" | null;

const FALLBACK_VERSION = "(dev)";

// The release version whose update toast the user has already closed.
// Stores the version rather than a boolean so dismissing 7.0.5 says
// nothing about 7.0.6: someone who cannot install updates on a managed
// machine is not nagged every launch, and everybody still hears about
// the next release.
const UPDATE_TOAST_DISMISSED_VERSION = "wsp.update-toast-dismissed-version";

const UPDATE_TOAST_ID = "update-available";

export function MenuCommands() {
  const navigate = useNavigate();
  const [dialog, setDialog] = useState<DialogId>(null);
  const [version, setVersion] = useState<string>(FALLBACK_VERSION);

  // Cached once; "Open user data folder" needs an absolute path that
  // varies per OS.
  const { data: setupStatus } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    staleTime: Infinity,
  });

  // App version for the Check-for-updates dialog.
  useEffect(() => {
    window.electronAPI
      ?.getVersion?.()
      .then(setVersion)
      .catch(() => setVersion("(unknown)"));
  }, []);

  // Dev-only: these dialogs are normally opened from the Electron native
  // menu, which doesn't exist in the browser dev server. Open one straight
  // from the URL hash so it can be previewed on localhost, e.g.
  // http://localhost:5173/#restore (also #backup, #reset, #updates, #model-library).
  // Tree-shaken out of production builds (import.meta.env.DEV is false).
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const openFromHash = () => {
      const h = window.location.hash.replace("#", "");
      if (
        h === "restore" ||
        h === "backup" ||
        h === "reset" ||
        h === "updates" ||
        h === "model-library"
      ) {
        setDialog(h as DialogId);
      }
    };
    openFromHash();
    window.addEventListener("hashchange", openFromHash);
    return () => window.removeEventListener("hashchange", openFromHash);
  }, []);

  // Check for a newer release once per launch. Until this existed the
  // only way to find out was the Help menu item, so nobody ever did:
  // installs sat two releases behind while a fixed bug was still biting
  // them. One request per launch is not polling, so GitHub's
  // unauthenticated rate limit is not a concern.
  //
  // Gated on setup being ready so it never lands on the first-run wizard
  // (never on top of the first-run wizard), and on the version parsing,
  // which skips the browser dev server where it is "(dev)" and any
  // build whose getVersion() failed. Failure and being up to date are
  // both silent: neither is news.
  const versionIsReal = parseVersion(version) !== null;
  const { latest, updateAvailable } = useLatestRelease(
    version,
    Boolean(setupStatus?.ready) && versionIsReal,
  );

  // A ref rather than dependency identity: a background
  // refetch must not resurrect a toast the user just closed.
  const updateToastShown = useRef(false);

  useEffect(() => {
    if (updateToastShown.current || !updateAvailable || !latest) return;
    if (localStorage.getItem(UPDATE_TOAST_DISMISSED_VERSION) === latest) return;
    updateToastShown.current = true;

    // A toast rather than the dialog: an update is worth mentioning, not
    // worth blocking the app on. Closing it is the dismissal, so it
    // never expires on its own; an update the user blinked past would
    // otherwise be gone until the next release.
    const remember = () =>
      localStorage.setItem(UPDATE_TOAST_DISMISSED_VERSION, latest);

    toast.info(`WSP CameraTrap ${formatVersion(latest)} is available`, {
      id: UPDATE_TOAST_ID,
      description: `You are running ${formatVersion(version)}.`,
      duration: Infinity,
      onDismiss: remember,
      action: {
        label: "Details",
        onClick: () => {
          remember();
          toast.dismiss(UPDATE_TOAST_ID);
          setDialog("updates");
        },
      },
    });
  }, [updateAvailable, latest, version]);

  // Reflect the stored species-name mode in the menu radio. Sent on mount;
  // a change reloads the page, so the next mount re-syncs the checkmark.
  useEffect(() => {
    window.electronAPI?.setSpeciesNameMenuMode?.(getSpeciesNameMode());
  }, []);

  // Gate the setup-only menu items on the wizard finishing. The query
  // shares its cache key with SetupGate (which polls), so `ready` flips to
  // true once setup completes and this re-runs to enable the items.
  useEffect(() => {
    window.electronAPI?.setMenuSetupReady?.(Boolean(setupStatus?.ready));
  }, [setupStatus?.ready]);

  // Dispatch menu commands. Re-subscribes when setupStatus loads so the
  // folder-open handler has the resolved path.
  useEffect(() => {
    const api = window.electronAPI;
    if (!api?.onMenuCommand) return;

    const openUserDataFolder = async () => {
      if (!setupStatus?.user_data_dir) {
        toast.error("User data path is unknown.");
        return;
      }
      const err = await api.openPath(setupStatus.user_data_dir);
      if (err) toast.error(`Could not open folder: ${err}`);
    };

    const openBackupsFolder = async () => {
      try {
        const { path } = await backupApi.getDir();
        const err = await api.openPath(path);
        if (err) toast.error(`Could not open folder: ${err}`);
      } catch (err) {
        toast.error(`Could not locate backups folder: ${(err as Error).message}`);
      }
    };

    return api.onMenuCommand((id) => {
      switch (id) {
        case "nav-home":
          navigate("/");
          break;
        case "new-project":
          // ?new=1 tells ProjectsPage to open the create dialog on arrival.
          navigate("/projects?new=1");
          break;
        case "new-folder-run":
          navigate("/folder-runs/new");
          break;
        case "about":
          navigate("/about");
          break;
        case "check-updates":
          setDialog("updates");
          break;
        case "backup":
          setDialog("backup");
          break;
        case "restore":
          setDialog("restore");
          break;
        case "reset":
          setDialog("reset");
          break;
        case "model-library": // WSP
          setDialog("model-library");
          break;
        case "open-user-data":
          void openUserDataFolder();
          break;
        case "open-backups":
          void openBackupsFolder();
          break;
        case "export-diagnostic":
          void exportDiagnosticReport();
          break;
        case "species-common":
          setSpeciesNameMode("common");
          break;
        case "species-scientific":
          setSpeciesNameMode("scientific");
          break;
      }
    });
  }, [navigate, setupStatus]);

  return (
    <>
      <ResetAppDialog
        open={dialog === "reset"}
        onOpenChange={(o) => setDialog(o ? "reset" : null)}
      />
      <BackupNowDialog
        open={dialog === "backup"}
        onOpenChange={(o) => setDialog(o ? "backup" : null)}
      />
      <RestoreBackupDialog
        open={dialog === "restore"}
        onOpenChange={(o) => setDialog(o ? "restore" : null)}
      />
      <CheckForUpdatesDialog
        open={dialog === "updates"}
        onOpenChange={(o) => setDialog(o ? "updates" : null)}
        currentVersion={version}
      />
      <ModelLibraryDialog
        open={dialog === "model-library"}
        onOpenChange={(o) => setDialog(o ? "model-library" : null)}
      />
    </>
  );
}
