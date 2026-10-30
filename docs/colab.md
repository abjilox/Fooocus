# Fooocus auf Colab: einmal einrichten, danach aus dem Cache starten

Öffne [das Notebook aus deinem Repository](https://colab.research.google.com/github/abjilox/Fooocus/blob/better-colab/colab.ipynb) und wähle eine GPU-Laufzeit mit Python 3.13.

## Bedienung

**Beim ersten Mal:** Konfiguration ausführen, dann **Einrichtung / Cache erstellen**, anschließend **Start**.

**In späteren Sitzungen:** Nur **Konfiguration** und **Start** ausführen. Die Start-Zelle funktioniert auch auf einer frisch zugeteilten Maschine ohne vorherige lokale Zustandsdatei. Sie installiert keine Pakete und startet keinen Paketresolver. Bei fehlendem oder inkompatiblem Cache nennt sie die Einrichtung als nächsten Schritt.

Die Einrichtung verwendet einen passenden vorhandenen Cache ebenfalls wieder. Mit `REBUILD_CACHE=True` lässt er sich ausdrücklich neu erstellen. Das ist nach einer Änderung des Abhängigkeitsprofils oder relevanter Colab-Pakete nötig. Reine Änderungen am Anwendungscode erzwingen keinen Neuaufbau der Pakete.

## Tatsächliche Ausgangsumgebung

Das Profil wurde auf die vom Nutzer gemeldete T4-Sitzung ausgerichtet:

| Bestandteil | Vorhandene Version |
| --- | --- |
| Python | 3.13.16 |
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| NumPy | 2.1.3 |
| Transformers | 5.21.0 |
| Gradio | 6.30.0 |
| Tokenizers | 0.23.3 |

Diese Versionen werden weiterverwendet. Das frühere Profil hatte NumPy ab 2.4 verlangt und gleichzeitig Änderungen an NumPy verhindert; das konnte in dieser Sitzung nicht installiert werden. Das korrigierte Profil verwendet für die Hintergrundentfernung `rembg==2.0.67`, das keine Aktualisierung auf NumPy 2.3 oder neuer verlangt. Die übrigen Mindestversionen wurden auf die tatsächlich benötigten Schnittstellen begrenzt.

## Was einmalig geschieht

Die Einrichtung ermittelt mit einem Installations-Trockenlauf ausschließlich die fehlenden oder inkompatiblen Zusatzpakete. Vorhandenes PyTorch, torchvision, torchaudio, CUDA, Triton, NumPy, Gradio und Transformers sind gegen Austausch geschützt. Es werden ausschließlich fertige Wheels verwendet; ein eigener CUDA-Unterbau wird nicht kompiliert.

Nur die zusätzlich benötigten Pakete werden in ein separates lokales Verzeichnis installiert. Colabs Systempakete bleiben unangetastet. Danach werden direkte und indirekte Abhängigkeiten, Imports und die Tests des Repositories geprüft. Erst nach erfolgreicher Prüfung werden Python-Dateien vorkompiliert und die Zusatzpakete als komprimiertes Archiv gespeichert.

Der private Cache liegt unter `MyDrive/Fooocus/cache/python`. Manifest und Prüfsumme halten fest, zu welcher Python-/Plattformumgebung, welchem Abhängigkeitsprofil und welchen tatsächlich verwendeten Colab-Paketen er passt. Modellgewichte, PyTorch und CUDA gehören nicht in dieses Paketarchiv. GitHub enthält weiterhin den Quellcode und das Profil.

Ein neues Archiv erhält einen unveränderlichen Namen. Das Manifest wird erst am Ende umgeschaltet; ein abgebrochener Neuaufbau ersetzt keinen funktionierenden Cache durch eine unvollständige Datei.

## Was beim normalen Start geschieht

In einer neuen Sitzung wird genau ein passendes Paketarchiv lokal übertragen, anhand seiner Prüfsumme geprüft und nach `/content/fooocus-deps` entpackt. Fooocus bindet dieses Verzeichnis vor den Systempaketen ein. Pakete werden nicht direkt aus vielen einzelnen Drive-Dateien importiert. Ist die Umgebung bereits lokal bereit, entfallen auch Übertragung und Entpacken.

Unterprozessausgaben werden fortlaufend in der Notebook-Zelle angezeigt. Ein Fehler zeigt damit seine eigentliche Ursache statt nur `CalledProcessError`. Beim Unterbrechen der Zelle werden auch die gestarteten Anwendungsprozesse beendet.

Die verbleibende Startzeit hängt insbesondere vom Übertragen und Laden der gewählten Modelle sowie vom Gradio-Freigabedienst ab. Eine feste Zeitgarantie wird nicht behauptet.

## Modelle und Ergebnisse

Programm und aktive Modelle liegen lokal unter `/content/Fooocus`. Drive hält Modelle dauerhaft in `MyDrive/Fooocus/models`. Eigene Checkpoints gehören nach `models/checkpoints`, LoRAs nach `models/loras`, Text-Embeddings nach `models/embeddings`. Modelle werden aufgelistet, aber erst bei Verwendung lokal kopiert. Bereits vorhandene lokale Modelle werden wiederverwendet.

Neue Downloads und Kopien werden zunächst als unvollständig behandelt und erst nach erfolgreichem Abschluss freigegeben. Fertige Bilder, eingebettete Metadaten und HTML-Protokolle werden nach `MyDrive/Fooocus/outputs` gesichert. Bei einem Drive-Schreibfehler bleibt die lokale Ausgabe erhalten; vor dem Beenden der Laufzeit muss ihre Sicherung geprüft werden.

Vorhandene Drive-Programmkopien werden nicht gelöscht. Ihre Konfiguration wird beim ersten lokalen Setup übernommen; Modell- und Ausgabepfade werden für die neue Sitzung lokal gesetzt. Konfiguration und Hash-Cache werden wieder auf Drive gesichert.

Ohne Drive bleiben Paket-Cache, Modelle und Ausgaben nur bis zum Löschen der Colab-Laufzeit erhalten.

## Prüfung

Lokale Tests mit den Abhängigkeiten des zu prüfenden Profils starten:

```sh
python scripts/run_tests.py
```

Die Prüfung umfasst native NumPy-/Gradio-/Transformers-Versionen, vollständigen Oberflächenaufbau, Maskenübergaben, den tatsächlich gepatchten CLIP-Pfad sowie kleine BLIP-, GPT-2-, Sicherheits- und GroundingDINO-Modelle. Cache-Tests prüfen Wiederherstellung ohne Installer, lokale Wiederverwendung, geänderte Laufzeiten, beschädigte Archive, unterbrochene Veröffentlichung und Schutz bestehender Verzeichnisse. Prozessprüfungen kontrollieren sichtbare Ausgaben und das Beenden verschachtelter Prozesse.

Am 31. Oktober 2026 bestanden im Linux-Container mit Python 3.13, PyTorch 2.11 CPU, NumPy 2.1.3, Transformers 5.21 und Gradio 6.30 alle **59 Tests**. Ein zusätzlicher vollständiger Cache-Durchlauf löste nur die zwei dort fehlenden Pakete auf, installierte sie getrennt, prüfte die Umgebung, veröffentlichte das Archiv und importierte es nach einer frischen Wiederherstellung erfolgreich. Kein Torch-/CUDA-/NumPy-/Gradio-/Transformers-Paket wurde dafür installiert.

Die Paket-Einrichtung einschließlich Tests dauerte in dieser vorbereiteten Testumgebung rund 68 Sekunden; die Wiederherstellung vom simulierten Drive-Verzeichnis rund 4 Sekunden, lokale Wiederverwendung etwa 0,2 Sekunden. Das sind lokale Linux-Messwerte für den Paket-Cache, keine zugesagte Colab-Gesamtstartzeit und keine Messung einer echten Drive-Verbindung. Die bisherige lokale Umgebung bestand ebenfalls die Tests; vier Prüfungen für moderne Bibliotheken wurden dort übersprungen.

Die unabhängige Codeprüfung und Linux-CPU-Tests ersetzen keine vollständige Colab-GPU-Abnahme. Auf der echten T4 sind weiterhin zu messen: einmalige Einrichtung, neue Sitzung mit Drive-Cache, erneuter Start derselben Sitzung und erstes Bild mit vorhandenen großen Modellgewichten.

Die Konsole zeigt Cache-Übertragung und Entpacken getrennt sowie die Phasen `environment_ready`, `models_ready`, `ui_initialized` und `first_image_saved`. Zeitmessungen stehen unter `/content/fooocus-colab-state.timings.jsonl`; die Einrichtung schreibt die Paketstände nach `/content/fooocus-colab-state.packages.json`.
