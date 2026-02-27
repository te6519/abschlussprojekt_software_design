# 2D Topologieoptimierer für Fachwerke

## Einleitung

Dieses Projekt implementiert einen 2D-Topologieoptimierer, der basierend auf der **Ground Structure Methode** optimale Fachwerkstrukturen ermittelt. Die Anwendung bietet ein interaktives **Streamlit-Webinterface**, in dem Nutzer Lastfälle definieren, Strukturen zeichnen und den Optimierungsprozess live verfolgen können.

Hier geht es zur Live-Demo: (https://abschlussprojektsoftwaredesignelimaxi.streamlit.app/)

## Beschreibung der Methode

Der Optimierer verwendet ein **"Hard-Kill"-Verfahren** auf einem diskreten Fachwerk. Dabei wird iterativ Material entfernt, das nur geringfügig zur Steifigkeit der Gesamtstruktur beiträgt.

Der iterative Zyklus besteht aus drei Schritten:
1.  **FEM-Berechnung:** Lösung des linearen Gleichungssystems $K \cdot u = F$ mittels der Direkten Steifigkeitsmethode, um Verschiebungen und Kräfte im Netzwerk zu bestimmen.
2.  **Energie-Bewertung:** Berechnung der Dehnungsenergie für jedes Element.
3.  **Reduktion:** Der Knoten mit der geringsten, gewichteten Gesamtenergie (und damit die an ihn angeschlossenen Stäbe) wird aus dem System entfernt.

## Anforderungen

Die Software setzt **Python 3** voraus und nutzt folgende essenzielle Bibliotheken:
*   `numpy`: Performante numerische Berechnungen.
*   `streamlit`: Aufbau der Web-Benutzeroberfläche.
*   `networkx`: Graphentheorie für Konnektivitäts- und Stabilitätsprüfungen.
*   `matplotlib`: Visualisierung der Ergebnisse und Heatmaps.
*   `pillow`: Bildverarbeitung für den Import von Strukturen.
*   `streamlit-drawable-canvas`: Zeichnen in der UI

## Installation & Deployment

### Lokale Installation

1.  Installieren Sie die Abhängigkeiten:
    ```bash
    pip install -r requirements.txt
    ```
2.  Starten Sie die Anwendung:
    ```bash
    streamlit run UI.py
    ```

### Cloud Deployment
Die Anwendung ist vollständig kompatibel mit der **Streamlit Community Cloud** und kann direkt aus dem GitHub-Repository heraus deployed werden.

## Nutzung

Der Workflow in der Benutzeroberfläche gliedert sich in 5 Schritte:

1.  **Modusauswahl:** Wählen Sie zwischen "Symmetrisch" (performante MBB-Balken-Optimierung) und "Universell" (für freie Geometrien).
2.  **Skizzieren/Importieren:** Zeichnen Sie den Bauraum direkt im Browser oder laden Sie ein Bild hoch.
3.  **Randbedingungen & Lasten:** Definieren Sie externe Kräfte und Lagerpunkte.
4.  **Optimierung starten:** Verfolgen Sie live, wie ineffizientes Material entfernt wird.
5.  **Export:** Speichern Sie das Ergebnis als Bild, Animation (GIF) oder Datensatz (JSON).

## Projektstruktur

Das Projekt trennt strikt zwischen Darstellung und Logik:

*   **`UI.py` (Frontend):** Behandelt Benutzereingaben, Interaktion mit dem Canvas und die Visualisierung der Plots.
*   **`system.py` (Backend):** Enthält die Kernlogik in den Klassen:
    *   `System`: Verwaltung des Gesamtzustands und FEM-Löser.
    *   `Node`: Repräsentation der Knotenpunkte.
    *   `Spring`: Federelemente und Berechnung der lokalen Steifigkeitsmatrizen.

![UML Diagram](UML-Diagram.png)

## Erfüllung der Anforderungen

### Minimalanforderungen
Die folgenden Pflichtkriterien wurden vollständig umgesetzt:

- [x] **Python-Anwendung mit Web-UI (Streamlit)**
- [x] **Topologieoptimierung beliebiger 2D-Strukturen**
- [x] **Definition der Ausgangsstruktur** (Bauraum, Randbedingungen, Externe Kräfte)
- [x] **Visualisierung** (Vorher, Nachher, Verformung)
- [x] **Speichern & Laden** des Projektzustands
- [x] **Lösung mittels FEM** (Anlehnung an Finite Elemente Methode)
- [x] **Stabilitätsverifikation** (Struktur fällt nicht auseinander)
- [x] **Bild-Download** der optimierten Geometrie

### Implementierte Erweiterungen
Zusätzlich zu den Minimalanforderungen wurden folgende Erweiterungen implementiert:

1.  **Hochladen eines Bildes (png, jpg):** Schwarze Pixel werden als Material interpretiert, weiße als Leerraum. Die Funktion `create_from_image` in `system.py` erzeugt daraus automatisch ein Knotengitter mit Federn. So lassen sich beliebige Bauraumgeometrien schnell definieren.
2.  **Zeichnen der Struktur im UI:** Über `streamlit-drawable-canvas` kann der Bauraum direkt im Browser per Freihandzeichnung skizziert werden — ohne externen Bildeditor.
3.  **Auszeichnungssprache (JSON) zum Speichern & Laden:** Der gesamte Systemzustand (Knoten, Federn, Kräfte, Randbedingungen) wird als JSON-Datei serialisiert (`save_to_dict` / `load_from_dict`). Dies ermöglicht das Herunterladen, erneute Hochladen und Fortsetzen einer unterbrochenen Optimierung.
4.  **Visualisierung des Optimierungskriteriums als Heatmap:** Die Dehnungsenergie jedes Stabs wird farblich kodiert (logarithmische Skalierung, Colormap `jet`). So ist auf einen Blick erkennbar, welche Bereiche tragend sind und welche entfernt werden können.
5.  **Animation als GIF speichern:** Der gesamte Optimierungsverlauf wird frameweise aufgezeichnet und als animiertes GIF exportiert. Die Schrittweite ist über den Slider "Jeden n-ten Schritt" konfigurierbar.

Darüber hinaus wurden folgende **algorithmische Erweiterungen** implementiert:

7.  **Sensitivitätsfilter:** Ein geometrischer Nachbarschaftsfilter (adaptiert nach Sigmund, 2001) verhindert isolierte "Spinnweben-Strukturen", indem die Energiedichte im Umfeld eines Knotens berücksichtigt wird.
8.  **Erweiterte Stabilitätsprüfung:**
    *   **Pfad-Check:** Stellt mittels `networkx` sicher, dass immer eine Verbindung zwischen Kraftangriffspunkt und Lager besteht.
    *   **Kinematik-Check:** Verhindert instabile Knotenketten, indem eine minimale Anbindungszahl (Grad $\ge 3$) gefordert wird.
9.  **Berechnungsmodi:**
    *   *Symmetrisch (MBB-Halbmodell):* Nutzt Symmetrieeigenschaften für schnellere Berechnung und spiegelt das Ergebnis zur Gesamtdarstellung.
    *   *Universell (Vollmodell):* Erlaubt völlig freie Gestaltung ohne Symmetriezwang mit frei wählbaren Lager- und Kraftpositionen.

## Quellen & Referenzen

*   **FEM und Direct Stiffness Method:** Implementiert gemäß Standard-Lehrliteratur zur Finiten-Elemente-Methode.
*   **Sensitivitätsfilter:** Adaptiert nach *Sigmund, O. (2001). A 99 line topology optimization code written in Matlab*.
*   **Bibliotheken:** Dokumentationen von [NetworkX](https://networkx.org/) und [Matplotlib](https://matplotlib.org/).
