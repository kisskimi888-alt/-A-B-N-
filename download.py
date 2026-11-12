"""Download the Hillstrom MineThatData e-mail experiment and verify its checksum."""
import hashlib
import pathlib
import urllib.request

URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
DEST = pathlib.Path(__file__).resolve().parent / "hillstrom.csv"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not DEST.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, DEST)
    digest = sha256(DEST)
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: got {digest}, expected {SHA256}")
    print(f"OK: {DEST} ({DEST.stat().st_size:,} bytes, sha256 verified)")


if __name__ == "__main__":
    main()
