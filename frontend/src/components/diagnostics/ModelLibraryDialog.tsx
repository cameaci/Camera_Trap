/**
 * Connect the WSP model library.
 *
 * The library is where models are installed from. It is either a
 * OneDrive/SharePoint share link to the library .zip (downloaded whenever
 * the file behind the link changes) or a folder: the synced
 * "WSP CameraTrap/models" OneDrive folder, found automatically, or a
 * network share. Saving re-reads the catalog, so newly published models
 * appear straight away.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderOpen, Library, Link2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { wspApi, type LibraryStatus } from "../../api/wsp";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Progress } from "../ui/progress";

interface ModelLibraryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const SOURCE_LABELS: Record<string, string> = {
  env: "set by WSP_MODEL_LIBRARY_DIR",
  settings: "folder chosen in this dialog",
  link: "downloaded from the OneDrive link",
  autodetect: "synced OneDrive folder, found automatically",
};

export function ModelLibraryDialog({ open, onOpenChange }: ModelLibraryDialogProps) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  // null until the user types: until then the field shows the link in use.
  const [linkDraft, setLinkDraft] = useState<string | null>(null);

  const { data: status, isLoading } = useQuery({
    queryKey: ["wsp-library"],
    queryFn: wspApi.getLibrary,
    enabled: open,
    // Poll while a download runs so the progress bar moves.
    refetchInterval: (query) =>
      (query.state.data as LibraryStatus | undefined)?.download_in_progress ? 1000 : false,
  });

  const link = linkDraft ?? status?.library_url ?? "";

  // When a download finishes, every model picker needs the new catalog.
  const wasDownloading = useRef(false);
  useEffect(() => {
    if (status?.download_in_progress) {
      wasDownloading.current = true;
    } else if (wasDownloading.current && status) {
      wasDownloading.current = false;
      void queryClient.invalidateQueries();
      if (!status.download_error) toast.success("WSP model library downloaded");
    }
  }, [status, queryClient]);

  const shownError =
    error ?? (status && !status.download_in_progress ? status.download_error : null);

  const apply = useMutation({
    mutationFn: (action: () => Promise<LibraryStatus>) => action(),
    onMutate: () => setError(null),
    onSuccess: (next) => {
      queryClient.setQueryData(["wsp-library"], next);
      if (!next.download_in_progress) {
        void queryClient.invalidateQueries();
        toast.success(
          next.library_dir
            ? "Model library connected"
            : "No model library found. Installed models keep working.",
        );
      }
    },
    onError: (err: Error) => setError(err.message),
  });

  const chooseFolder = async () => {
    const dir = await window.electronAPI?.selectFolder?.();
    if (dir) apply.mutate(() => wspApi.setLibraryDir(dir));
  };

  const busy = apply.isPending || Boolean(status?.download_in_progress);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Library className="h-5 w-5" />
            WSP model library
          </DialogTitle>
          <DialogDescription>
            Models are installed from the WSP model library on OneDrive. Paste
            the share link of the library .zip, or use the synced
            "WSP CameraTrap/models" folder.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 text-sm">
          <div className="rounded-md border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Library in use
            </div>
            {isLoading ? (
              <div className="mt-1 text-muted-foreground">Checking…</div>
            ) : status?.library_dir ? (
              <>
                <div className="mt-1 break-all font-mono text-xs">{status.library_dir}</div>
                {status.source && (
                  <div className="mt-1 text-xs text-muted-foreground">
                    {SOURCE_LABELS[status.source] ?? status.source}
                  </div>
                )}
              </>
            ) : (
              <div className="mt-1 text-amber-700">
                {status?.configured_dir
                  ? `The saved folder cannot be reached: ${status.configured_dir}. Is OneDrive synced?`
                  : "No library connected yet."}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="wsp-library-link">OneDrive share link</Label>
            <div className="flex gap-2">
              <Input
                id="wsp-library-link"
                placeholder="https://wsponline-my.sharepoint.com/:u:/…"
                value={link}
                onChange={(e) => setLinkDraft(e.target.value)}
                disabled={busy}
              />
              <Button
                disabled={busy || !link.trim()}
                onClick={() => apply.mutate(() => wspApi.setLibraryUrl(link.trim()))}
              >
                <Link2 className="mr-2 h-4 w-4" />
                Connect
              </Button>
            </div>
            {status?.download_in_progress && (
              <div className="space-y-1">
                <Progress value={Math.round((status.download_progress || 0) * 100)} />
                <div className="text-xs text-muted-foreground">
                  {status.download_message || "Downloading the WSP model library…"}
                </div>
              </div>
            )}
          </div>

          {status && (
            <p className="text-xs text-muted-foreground">
              Installed models are kept in{" "}
              <span className="break-all font-mono">{status.models_dir}</span>. A model
              folder copied there by hand works too.
            </p>
          )}
          {shownError && <p className="text-sm text-destructive">{shownError}</p>}
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <div className="flex gap-2">
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => {
                setLinkDraft(null);
                apply.mutate(() => wspApi.resetLibrary());
              }}
            >
              Reset
            </Button>
            {status?.library_url && (
              <Button
                variant="ghost"
                disabled={busy}
                onClick={() => apply.mutate(() => wspApi.refreshLibrary())}
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Check for new models
              </Button>
            )}
          </div>
          <Button variant="outline" disabled={busy} onClick={() => void chooseFolder()}>
            <FolderOpen className="mr-2 h-4 w-4" />
            Use a folder…
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
