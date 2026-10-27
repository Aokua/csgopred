"""HLTV 解析器测试（纯离线，用最小 HTML 片段固定结构假设）。

页面结构会随 HLTV 改版变化；这些测试的作用是：改版后失败能立刻定位到选择器，
而不是悄悄产出空数据。
"""

from csgopred.ingest.hltv import parse_match, parse_results

RESULTS_HTML = """
<html><body>
<div class="results-sublist">
  <div class="result-con " data-zonedgrouping-entry-unix="1793023511000">
    <a href="/matches/2399738/9z-vs-tyloo-pgl-masters-bucharest-2026" class="a-reset">
      <div class="result"><table><tr>
        <td class="team-cell"><div class="line-align team1"><div class="team ">9z</div></div></td>
        <td class="result-score"><span class="score-lost">1</span> - <span class="score-won">2</span></td>
        <td class="team-cell"><div class="line-align team2"><div class="team team-won">TYLOO</div></div></td>
        <td class="event"><span class="event-name">PGL Masters Bucharest 2026</span></td>
        <td class="star-cell"><div class="map-text">bo3</div></td>
      </tr></table></div>
    </a>
  </div>
  <div class="result-con " data-zonedgrouping-entry-unix="1792835802000">
    <a href="/matches/2399589/lynn-vision-vs-chinggis-warriors" class="a-reset">
      <div class="result"><table><tr>
        <td class="team-cell"><div class="team team-won">Lynn Vision</div></td>
        <td class="result-score"><span class="score-won">3</span> - <span class="score-lost">0</span></td>
        <td class="team-cell"><div class="team ">Chinggis Warriors</div></td>
        <td class="event"><span class="event-name">MaceMedia VOID Autumn Cup 2026</span></td>
        <td class="star-cell"><div class="map-text">bo5</div></td>
      </tr></table></div>
    </a>
  </div>
</div>
</body></html>
"""

MATCH_HTML = """
<html><body>
<div class="teamName">Vitality</div>
<div class="teamName">FUT</div>
<div class="veto-box"><div class="padding preformatted-text">Best of 3 (LAN)</div></div>
<div class="veto-box"><div class="padding">
  <div>1. FUT removed Inferno</div>
  <div>2. Vitality removed Ancient</div>
  <div>3. FUT picked Mirage</div>
  <div>4. Vitality picked Dust2</div>
  <div>5. Vitality removed Nuke</div>
  <div>6. FUT removed Cache</div>
  <div>7. Anubis was left over</div>
</div></div>
<div class="mapholder"><div class="played"><div class="mapname">Mirage</div></div>
  <div class="results played">
    <div class="results-left won"><div class="results-team-score">13</div></div>
    <span class="results-right lost"><div class="results-team-score">9</div></span>
  </div>
</div>
<div class="mapholder"><div class="played"><div class="mapname">Dust2</div></div>
  <div class="results played">
    <div class="results-left lost"><div class="results-team-score">9</div></div>
    <span class="results-right won"><div class="results-team-score">13</div></span>
  </div>
</div>
<div class="mapholder"><div class="played"><div class="mapname">Default</div></div>
  <div class="results played">
    <div class="results-left won"><div class="results-team-score">0</div></div>
    <span class="results-right lost"><div class="results-team-score">1</div></span>
  </div>
</div>
</body></html>
"""


def test_parse_results_fields():
    rows = parse_results(RESULTS_HTML)
    assert len(rows) == 2
    r = rows[0]
    assert r["match_id"] == "2399738"
    assert r["slug"].startswith("9z-vs-tyloo")
    assert (r["team_a"], r["team_b"]) == ("9z", "TYLOO")
    assert (r["score_a"], r["score_b"]) == (1, 2)
    assert r["format"] == "bo3"
    assert r["date"] == "2026-10-26"


def test_parse_results_handles_bo5():
    rows = parse_results(RESULTS_HTML)
    assert rows[1]["format"] == "bo5"
    assert (rows[1]["score_a"], rows[1]["score_b"]) == (3, 0)


def test_parse_match_teams_and_veto():
    d = parse_match(MATCH_HTML)
    assert d["teams"] == ["Vitality", "FUT"]
    assert len(d["veto"]) == 7
    assert d["veto"][0] == {"order": 1, "team": "FUT", "action": "removed", "map": "Inferno"}
    assert d["veto"][2]["action"] == "picked"
    assert d["veto"][-1]["action"] == "left over"


def test_parse_match_drops_placeholder_maps():
    d = parse_match(MATCH_HTML)
    assert [m["map"] for m in d["maps"]] == ["Mirage", "Dust2"]
    assert d["maps"][0]["a_won"] is True
    assert d["maps"][1]["a_won"] is False
