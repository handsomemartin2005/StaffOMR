import tempfile
import unittest
from pathlib import Path

from tools.sanitize_musicxml_chords import sanitize_musicxml_chords


class SanitizeMusicXmlChordsTest(unittest.TestCase):
    def test_removes_orphan_chords_but_keeps_valid_chord_members(self):
        xml = """<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part id="P1"><measure number="1">
    <note><chord/><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
    <note><rest/><duration>4</duration></note>
    <note><chord/><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration></note>
    <note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration></note>
  </measure></part>
</score-partwise>
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.musicxml"
            path.write_text(xml, encoding="utf-8")

            removed = sanitize_musicxml_chords(path)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(removed, 2)
        self.assertEqual(text.count("<chord"), 1)


if __name__ == "__main__":
    unittest.main()
