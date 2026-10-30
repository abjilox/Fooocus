# Fooocus auf Google Colab

Öffne [das Notebook aus deinem Repository](https://colab.research.google.com/github/abjilox/Fooocus/blob/better-colab/colab.ipynb). Wähle unter **Laufzeit → Laufzeittyp ändern** eine GPU und die aktuelle Python-3.13-Laufzeit. Führe zuerst **Setup**, danach **Start** aus. Für einen Neustart in derselben Sitzung genügt die Start-Zelle.

Die Anpassung verwendet Colabs vorhandenes Python und GPU-PyTorch. Die geprüften Zusatzabhängigkeiten stehen in `requirements_colab.txt`; die Anforderungen für bisherige lokale Installationen bleiben in `requirements_versions.txt`. Der Colab-Start lädt diese alte Datei nicht nach. Die tatsächlichen Paketversionen werden beim Setup ausgegeben. Eine fehlgeschlagene Installation beendet das Setup.

## Referenz und Grenzen

Referenz für die Anpassung ist Googles veröffentlichter GPU-Schnappschuss vom Oktober 2026:

| Bestandteil | Bisheriges Fooocus-Profil | Colab-Referenz |
| --- | --- | --- |
| Python | Dokumentation: 3.10 | 3.13.16 |
| PyTorch | Installation bei Bedarf: 2.1.0 | 2.11.0+cu130 |
| torchvision | Installation bei Bedarf: 0.16.0 | 0.26.0+cu130 |
| NumPy | 1.26.4 | 2.4.6 |
| Transformers | 4.42.4 | 5.22.0 |
| Gradio | 3.41.2 | 6.31.0 |

Quellen: [Betriebssystem und Python](https://raw.githubusercontent.com/googlecolab/backend-info/main/os-info-gpu.txt), [GPU-Pakete](https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.gpu.txt), [Hinweise zur zeitversetzten Ausrollung](https://github.com/googlecolab/backend-info). Der veröffentlichte Stand ersetzt keine Prüfung der tatsächlich zugeteilten Sitzung. Es wird keine ältere Python-Laufzeit eingerichtet und kein älteres CUDA-Paket heruntergeladen.

**Die echte Colab-GPU-Abnahme ist noch offen.** Lokale Tests von Dateiverwaltung, Adaptern und kleinen Modellen ersetzen keine Bildgenerierung auf einer Colab-GPU. Die unten genannten Szenarien müssen dort durchlaufen, bevor diese Kombination als praktisch bestätigt gilt.

## Speicherorte

- Programm und Python-Pakete bleiben auf der lokalen Colab-Festplatte. Eine vorhandene Programmkopie auf Drive wird nicht gelöscht oder aktualisiert.
- Modelle liegen dauerhaft in `MyDrive/Fooocus/models`. Lege eigene Checkpoints unter `models/checkpoints`, LoRAs unter `models/loras` ab. Die Auswahl zeigt auch Drive-Modelle an; erst bei Verwendung wird eine lokale Kopie erstellt.
- Neue Modell-Downloads werden zuerst lokal abgeschlossen und danach auf Drive gesichert. Unvollständige Dateien tragen `.partial` und gelten nicht als fertige Modelle.
- Fertige Bilder mit eingebetteten Metadaten und das HTML-Protokoll werden unter `MyDrive/Fooocus/outputs` gesichert. Bei einem Drive-Fehler bleibt die lokale Datei erhalten; die Konsole nennt den Fehler und den lokalen Pfad. Vor dem Beenden der Laufzeit die Sicherung kontrollieren.
- Die vorhandene Drive-Konfiguration wird beim ersten Setup übernommen. Die Modell- und Ausgabepfade werden für die aktuelle Sitzung auf lokale Ordner gesetzt. Konfiguration und Hash-Cache werden wieder auf Drive gesichert.
- Ohne Drive bleibt die Anwendung nutzbar; lokale Modelle und Ausgaben gehen beim Löschen der Laufzeit verloren.

Für die Arbeitskopien wird ausreichend freier lokaler Speicher benötigt. Kopien prüfen den freien Platz vor dem Übertragen. Die erste Sitzung benötigt weiterhin Downloads; eine neue Sitzung benötigt die Übertragung der verwendeten Modelle von Drive. Bereits vorhandene lokale Dateien werden wiederverwendet.

## Entwicklung und Prüfungen

Die Entwicklung erfolgt auf `better-colab`. Das Notebook klont ausschließlich `abjilox/Fooocus` und aktualisiert nur per Fast-forward. Bei eigenen Änderungen oder einem anderen Repository bricht es ab, ohne Dateien zu überschreiben. Der Start führt kein automatisches Git-Update aus.

Lokale Tests aus dem Repository starten:

```sh
python scripts/run_tests.py
```

Der Teststarter verhindert, dass Fooocus die Kommandozeilenargumente des Testwerkzeugs als Anwendungsoptionen interpretiert. Benötigt wird eine Python-Umgebung mit den Abhängigkeiten des jeweils zu prüfenden Profils.

Validiert am 30. Oktober 2026:

- Linux-Container mit Python 3.13.18, PyTorch 2.11.0+cpu, torchvision 0.26.0+cpu, NumPy 2.4.6, Transformers 5.22.1 und Gradio 6.31.0: **40 Tests bestanden**, keine übersprungen. Das vollständige Colab-Abhängigkeitsprofil ließ sich dort installieren. Zusätzlich wurde die reine Wheel-Auflösung für Linux/Python 3.13 geprüft.
- Gradio 6.31.0: Oberfläche mit 336 Komponenten und 77 Ereignissen aufgebaut; öffentliche Dateirouten und Maskenübergaben geprüft. Die bisherige Gradio-3.41-Oberfläche lässt sich ebenfalls aufbauen.
- Kleine Offline-Modelle prüfen CLIP einschließlich des tatsächlich gepatchten Fooocus-Pfads und Checkpoint-Schlüsseln, BLIP-Textgenerierung, GPT-2-Gewichtsladen und Generierung, Sicherheitsprüfung sowie GroundingDINO ohne eigene CUDA-Kompilierung.
- Drive-Verhalten wurde mit lokalen temporären Ordnern geprüft: gezielte Modellkopien einschließlich FaceSwap und Text-Embeddings, Wiederverwendung, unterbrochene Übertragungen, Speicherknappheit und Erhalt lokaler Ausgaben bei Schreibfehlern.

Diese Prüfungen verwenden kleine Modelle und simulieren Drive über Verzeichnisse. Große Modellgewichte, echte Drive-Verbindungen, Browsergesten und CUDA-Bildgenerierung sind damit nicht abgenommen.

## Colab-Abnahme

In einer frisch zugeteilten aktuellen GPU-Laufzeit prüfen und die gedruckten Versionsstände festhalten:

1. Setup und Start; die Gradio-Freigabe öffnen. PyTorch, torchvision und CUDA-Pakete dürfen nicht ersetzt werden.
2. Mit dem Standardmodell ein Bild erzeugen, Prompt-Erweiterung und eine LoRA verwenden. Bild und Metadaten auf Drive kontrollieren.
3. Bildvorschlag, Variation/Upscaling, gezeichnete und hochgeladene Inpainting-Masken, automatische Masken, Bildbeschreibung und Metadatenimport ausprobieren.
4. Start erneut ausführen: keine Paketinstallation und keine vollständige Modellkopie für bereits lokal vorhandene Modelle.
5. Laufzeit löschen, neu verbinden und erneut starten: benötigte Modelle aus Drive übernehmen, andere Checkpoints noch nicht kopieren.
6. Einen Download beziehungsweise eine Drive-Kopie unterbrechen und erneut versuchen. Eine unvollständige Datei darf nicht als fertiges Modell geladen werden.

Die Konsole meldet `environment_ready`, `models_ready`, `ui_initialized` und `first_image_saved`. Zusätzlich schreibt der Start Zeitmessungen nach `/content/fooocus-colab-state.timings.jsonl`. Die installierten Paketversionen stehen nach dem Setup in `/content/fooocus-colab-state.packages.json`; die Zustandsdatei enthält auch den verwendeten Git-Commit. Setup einschließlich Git-Aktualisierung wird separat gemessen. Netzwerk- und GPU-Zuteilung variieren; es gibt keine feste Zeitgarantie.

| Messung | Status |
| --- | --- |
| Einrichtung einer frischen Colab-Sitzung | Noch nicht auf Colab gemessen |
| Oberfläche und erstes Bild | Noch nicht auf Colab gemessen |
| Wiederholter Start derselben Sitzung | Noch nicht auf Colab gemessen |
| Neue Sitzung mit vorhandenen Drive-Modellen | Noch nicht auf Colab gemessen |
