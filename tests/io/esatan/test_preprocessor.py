"""Tests for the ESATAN .d preprocessor (file IO, includes, sanitisation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pycanha.io.esatan.errors import EsatanParseError
from pycanha.io.esatan.preprocessor import (
    esatan_float,
    expand_includes,
    sanitise_d_notation,
    strip_data_comments,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestEsatanFloat:
    def test_plain_float(self) -> None:
        assert esatan_float("3.14") == pytest.approx(3.14)

    def test_fortran_d_notation(self) -> None:
        assert esatan_float("9.76D-06") == pytest.approx(9.76e-06)
        assert esatan_float("0.D+00") == 0.0
        assert esatan_float("5.0D0") == 5.0
        assert esatan_float("1d3") == 1000.0

    def test_scientific_notation(self) -> None:
        assert esatan_float("1.5e-3") == pytest.approx(0.0015)
        assert esatan_float("3.141593E-007") == pytest.approx(3.141593e-7)

    def test_blank_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            esatan_float("   ")


class TestSanitiseDNotation:
    def test_numeric_literal(self) -> None:
        assert sanitise_d_notation("9.76D-06 * 2") == "9.76e-06 * 2"

    def test_does_not_touch_identifiers(self) -> None:
        # ``D1`` is a node label, ``DAMPT`` is a control constant.
        assert sanitise_d_notation("D1") == "D1"
        assert sanitise_d_notation("DAMPT") == "DAMPT"
        assert sanitise_d_notation("DTIMEI = 5.0D0") == "DTIMEI = 5.0e0"

    def test_in_long_text(self) -> None:
        text = "C = 9.76D-06 * NODFN1(T1, Cp_DUT, 1) * 1800.0D0;"
        out = sanitise_d_notation(text)
        assert "9.76e-06" in out
        assert "1800.0e0" in out
        # Identifiers preserved.
        assert "NODFN1" in out
        assert "Cp_DUT" in out


class TestStripDataComments:
    def test_eol_comments(self) -> None:
        text = "D1 = 5;  # eol comment\nD2 = 6;\n"
        out = strip_data_comments(text)
        assert "# eol" not in out
        assert "D1 = 5;" in out
        assert "D2 = 6;" in out

    def test_fortran_column_one_c(self) -> None:
        text = "C this is a Fortran comment\nD1 = 5;\n"
        out = strip_data_comments(text)
        assert "Fortran comment" not in out
        assert "D1 = 5;" in out

    def test_hash_inside_string_is_data(self) -> None:
        # Node labels, node lists ('#1-100') and paths can contain '#'.
        text = "D5 = 'PANEL#5', T = 25.0;  # a note\nNGROUP = \"#2\";\n"
        out = strip_data_comments(text)
        assert "D5 = 'PANEL#5', T = 25.0;" in out
        assert "a note" not in out
        assert 'NGROUP = "#2";' in out


class TestExpandIncludes:
    def test_inlines_relative_includes(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub.dat"
        sub.write_text("$REAL\n  k = 0.5;\n", encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text(
            '$LOCALS\n  $INCLUDE "sub.dat"\n',
            encoding="utf-8",
        )

        out = expand_includes(main)
        assert "k = 0.5" in out
        assert "$INCLUDE" not in out

    def test_inlines_absolute_includes(self, tmp_path: Path) -> None:
        absolute = tmp_path / "absolute.dat"
        absolute.write_text("k_abs = 1.0;\n", encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text(
            f'$LOCALS\n  $INCLUDE "{absolute}"\n',
            encoding="utf-8",
        )

        out = expand_includes(main)
        assert "k_abs = 1.0" in out

    def test_recursive_includes(self, tmp_path: Path) -> None:
        a = tmp_path / "a.dat"
        b = tmp_path / "b.dat"
        a.write_text('$INCLUDE "b.dat"\n', encoding="utf-8")
        b.write_text("a = 1;\n", encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text('$INCLUDE "a.dat"\n', encoding="utf-8")

        out = expand_includes(main)
        assert "a = 1" in out

    def test_cycle_detection(self, tmp_path: Path) -> None:
        a = tmp_path / "a.dat"
        b = tmp_path / "b.dat"
        a.write_text('$INCLUDE "b.dat"\n', encoding="utf-8")
        b.write_text('$INCLUDE "a.dat"\n', encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text('$INCLUDE "a.dat"\n', encoding="utf-8")

        with pytest.raises(EsatanParseError, match="cycle"):
            expand_includes(main)

    @pytest.mark.parametrize(
        "line",
        [
            '#$INCLUDE "missing.dat"',
            '    # $INCLUDE "missing.dat"',
            'C     $INCLUDE "missing.dat"',
            'GL(1, 2) = 1.0;  # $INCLUDE "missing.dat"',
        ],
    )
    def test_commented_out_include_is_not_read(self, tmp_path: Path, line: str) -> None:
        main = tmp_path / "main.d"
        main.write_text(f"$CONDUCTORS\n{line}\n", encoding="utf-8")

        out = expand_includes(main)
        assert line in out

    def test_commented_out_include_of_existing_file(self, tmp_path: Path) -> None:
        # Inlined, the file would be commented out on its first line only.
        two = tmp_path / "two.dat"
        two.write_text("GL(1, 9) = 100.0;\nGL(2, 4) = 100.0;\n", encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text('$CONDUCTORS\n    #$INCLUDE "two.dat"\n', encoding="utf-8")

        out = expand_includes(main)
        assert "GL(2, 4)" not in out

    def test_include_path_with_hash(self, tmp_path: Path) -> None:
        sub = tmp_path / "case#1.dat"
        sub.write_text("k = 0.5;\n", encoding="utf-8")
        main = tmp_path / "main.d"
        main.write_text('$LOCALS\n  $INCLUDE "case#1.dat"  # case 1\n', encoding="utf-8")

        out = expand_includes(main)
        assert "k = 0.5" in out
