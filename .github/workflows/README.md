# GitHub Actions

| Workflow | When | What |
|---|---|---|
| `tests.yml` | every PR and push to `main` | frontend typecheck, backend lint/mypy/tests, WSP tool and training tests |
| `build-windows.yml` | a published release, or run by hand from the Actions tab | builds `WSP-CameraTrap-Setup-<version>.exe` and attaches it to the release |

Signing is optional: add the repository secrets `WSP_PFX_BASE64` (the
code-signing `.pfx`, base64-encoded) and `WSP_PFX_PASSWORD` to get a
signed installer. See `wsp/docs/ADMIN_GUIDE.md` for the release steps.
