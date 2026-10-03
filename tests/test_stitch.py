"""场景拼接（stitch）接口测试。

POST /stitch：按给定顺序把 2～4 个已保存场景拼成连续播放场景，保存为
revision=1 的新场景。覆盖时间平移与段序、id 重写、保轨、跨段 changes、
同源重复、404/409/422、严格整数、原子性、源与播放会话不变、目标可读取/
修订/播放，以及“源快照与核对版本错配”的可控交错回归。
"""
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.solver import Subtitle, solve  # noqa: E402


def make_scene(
    client,
    scene_id,
    subtitles,
    track_count=2,
    expected_revision=0,
):
    resp = client.put(
        f"/scenes/{scene_id}",
        json={
            "track_count": track_count,
            "expected_revision": expected_revision,
            "subtitles": subtitles,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def stitch(client, target, segments):
    return client.post(
        "/stitch",
        json={"target_scene_id": target, "segments": segments},
    )


def seg(scene_id, expected_revision=1, duration_ms=None):
    return {
        "scene_id": scene_id,
        "expected_revision": expected_revision,
        "duration_ms": duration_ms,
    }


def default_a(client, scene_id="scene-a"):
    # a1 Alice[0,1000) -> 轨1；a2 Bob[800,3000) 锁轨2；a3 Alice[1000,2500) -> 轨1
    # changes=0；max end_ms=3000
    return make_scene(
        client,
        scene_id,
        [
            {"id": "a1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "a2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "a3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ],
    )


def default_b(client, scene_id="scene-b", track_count=2):
    # b1 Carol[0,500)；b2 Alice[0,2000) -> b1 轨1、b2 轨2
    # （b2 与 b1 重叠，不能同轨）；max end_ms=2000
    return make_scene(
        client,
        scene_id,
        [
            {"id": "b1", "speaker": "Carol", "start_ms": 0, "end_ms": 500},
            {"id": "b2", "speaker": "Alice", "start_ms": 0, "end_ms": 2000},
        ],
        track_count=track_count,
    )


# ---------------------------------------------------------------------------
# 基本拼接语义
# ---------------------------------------------------------------------------


def test_stitch_two_segments_shifts_times_rewrites_ids_and_preserves_tracks():
    client = TestClient(app)
    a = default_a(client)
    b = default_b(client)

    resp = stitch(
        client,
        "scene-ab",
        [
            seg("scene-a", 1, 3000),
            seg("scene-b", 1, 2000),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["scene_id"] == "scene-ab"
    assert body["revision"] == 1
    assert body["track_count"] == 2
    assert body["feasible"] is True

    # 第二段整体平移 3000；id 为“段号:段内序号”（字符串），speaker 不变；
    # locked_track 为源实际轨道
    assert body["subtitles"] == [
        {"id": "1:1", "speaker": "Alice",
         "start_ms": 0, "end_ms": 1000, "locked_track": 1},
        {"id": "1:2", "speaker": "Bob",
         "start_ms": 800, "end_ms": 3000, "locked_track": 2},
        {"id": "1:3", "speaker": "Alice",
         "start_ms": 1000, "end_ms": 2500, "locked_track": 1},
        {"id": "2:1", "speaker": "Carol",
         "start_ms": 3000, "end_ms": 3500, "locked_track": 1},
        {"id": "2:2", "speaker": "Alice",
         "start_ms": 3000, "end_ms": 5000, "locked_track": 2},
    ]
    # 源画面轨道逐一保持
    assert body["assignments"] == [
        {"id": "1:1", "track": 1},
        {"id": "1:2", "track": 2},
        {"id": "1:3", "track": 1},
        {"id": "2:1", "track": 1},
        {"id": "2:2", "track": 2},
    ]

    # 源未被修改
    assert client.get("/scenes/scene-a").json() == a
    assert client.get("/scenes/scene-b").json() == b

    # 目标已持久化为 revision 1
    assert client.get("/scenes/scene-ab").json() == body


def test_stitch_changes_count_across_segment_boundary():
    """changes 按拼接后同一说话人时间相邻字幕计算（含跨段相邻）。"""
    client = TestClient(app)
    default_a(client)
    default_b(client)

    body = stitch(
        client,
        "scene-ab",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 2000)],
    ).json()

    # Alice 时间链：1:1(轨1) -> 1:3(轨1) -> 2:2(轨2)，恰一次换轨；
    # Bob、Carol 各只有一条，不计
    assert body["changes"] == 1


def test_stitch_assignments_and_changes_match_solver():
    """以拼接字幕（locked_track=源轨道）交求解器复算，结果必须一致。"""
    client = TestClient(app)
    default_a(client)
    default_b(client)

    body = stitch(
        client,
        "scene-ab",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 2000)],
    ).json()

    subs = [
        Subtitle(
            id=s["id"], speaker=s["speaker"], start_ms=s["start_ms"],
            end_ms=s["end_ms"], locked_track=s["locked_track"],
        )
        for s in body["subtitles"]
    ]
    tracks, changes = solve(subs, 2)
    assert [a["track"] for a in body["assignments"]] == tracks
    assert body["changes"] == changes


def test_stitch_three_and_four_segments_with_cumulative_offsets():
    client = TestClient(app)
    default_a(client)  # max end 3000
    default_b(client)  # max end 2000

    body = stitch(
        client,
        "scene-aaa",
        [
            seg("scene-a", 1, 3000),
            seg("scene-a", 1, 3000),
            seg("scene-a", 1, 3000),
        ],
    ).json()
    assert [s["id"] for s in body["subtitles"]] == [
        "1:1", "1:2", "1:3", "2:1", "2:2", "2:3", "3:1", "3:2", "3:3",
    ]
    # 每段整体平移 3000 的倍数
    assert [(s["start_ms"], s["end_ms"]) for s in body["subtitles"]] == [
        (0, 1000), (800, 3000), (1000, 2500),
        (3000, 4000), (3800, 6000), (4000, 5500),
        (6000, 7000), (6800, 9000), (7000, 8500),
    ]
    assert body["changes"] == 0  # 同源重复，轨1/2/1 模式不变

    # 四段上限（2 轨；3+3+3+3=12 条恰好不超限）
    body4 = stitch(
        client,
        "scene-aaaa",
        [seg("scene-a", 1, 3000) for _ in range(4)],
    )
    assert body4.status_code == 200
    assert len(body4.json()["subtitles"]) == 12


def test_stitch_same_source_repeated_independent_revisions():
    """同源可重复：每个出现位置独立给出并核对 expected_revision。"""
    client = TestClient(app)
    default_a(client)
    # 修订到 revision 2
    make_scene(
        client,
        "scene-a",
        [
            {"id": "only", "speaker": "Solo", "start_ms": 0, "end_ms": 50},
        ],
        expected_revision=1,
    )

    # 同一段序列里第一次按 revision 1 必然 409（源当前为 2）
    resp = stitch(
        client,
        "scene-x",
        [seg("scene-a", 1, 50), seg("scene-a", 2, 50)],
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["segment"] == 1
    assert detail["current_revision"] == 2

    # 两处都按 revision 2：成功，两段都取自 revision 2 快照
    body = stitch(
        client,
        "scene-y",
        [seg("scene-a", 2, 50), seg("scene-a", 2, 50)],
    ).json()
    assert [s["id"] for s in body["subtitles"]] == ["1:1", "2:1"]
    assert [s["speaker"] for s in body["subtitles"]] == ["Solo", "Solo"]
    assert [(s["start_ms"], s["end_ms"]) for s in body["subtitles"]] == [
        (0, 50), (50, 100)
    ]


def test_stitch_source_integer_ids_become_segment_style_string_ids():
    client = TestClient(app)
    make_scene(
        client,
        "scene-int",
        [
            {"id": 9, "speaker": "Zoe", "start_ms": 0, "end_ms": 100},
            {"id": 1, "speaker": "Abe", "start_ms": 50, "end_ms": 200},
        ],
    )
    body = stitch(
        client,
        "scene-int-out",
        [seg("scene-int", 1, 200), seg("scene-int", 1, 200)],
    ).json()
    # id 一律重写为“段号:段内序号”字符串，源顺序保持
    assert [s["id"] for s in body["subtitles"]] == [
        "1:1", "1:2", "2:1", "2:2",
    ]
    assert [s["speaker"] for s in body["subtitles"]] == [
        "Zoe", "Abe", "Zoe", "Abe",
    ]
    assert [a["id"] for a in body["assignments"]] == [
        "1:1", "1:2", "2:1", "2:2",
    ]


def test_stitch_duration_may_exceed_source_length_leaving_silence():
    """duration_ms 可以大于源最大 end_ms：段间留出静音间隔。"""
    client = TestClient(app)
    default_a(client)
    default_b(client)

    body = stitch(
        client,
        "scene-gap",
        [seg("scene-a", 1, 5000), seg("scene-b", 1, 2000)],
    ).json()
    # 第二段从 5000 起，与第一段最后结束（3000）之间留 2000 空档
    assert body["subtitles"][3]["start_ms"] == 5000
    assert body["subtitles"][3]["end_ms"] == 5500
    assert body["subtitles"][4]["end_ms"] == 7000


def test_stitch_duration_equal_to_max_end_ms_is_allowed():
    client = TestClient(app)
    default_a(client)
    body = stitch(
        client,
        "scene-eq",
        [seg("scene-a", 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert body.status_code == 200


def test_stitch_preserves_source_tracks_with_three_tracks():
    client = TestClient(app)
    s1 = make_scene(
        client,
        "scene-3a",
        [
            {"id": "x", "speaker": "X", "start_ms": 0, "end_ms": 100,
             "locked_track": 3},
        ],
        track_count=3,
    )
    s2 = make_scene(
        client,
        "scene-3b",
        [
            {"id": "y", "speaker": "Y", "start_ms": 0, "end_ms": 100,
             "locked_track": 2},
        ],
        track_count=3,
    )
    body = stitch(
        client,
        "scene-3out",
        [seg("scene-3a", 1, 100), seg("scene-3b", 1, 100)],
    ).json()
    assert body["track_count"] == 3
    assert body["subtitles"][0]["locked_track"] == 3
    assert body["subtitles"][1]["locked_track"] == 2
    assert [a["track"] for a in body["assignments"]] == [3, 2]
    assert client.get("/scenes/scene-3a").json() == s1
    assert client.get("/scenes/scene-3b").json() == s2


# ---------------------------------------------------------------------------
# 422：非法请求
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 5, 6])
def test_stitch_segment_count_must_be_2_to_4(count):
    client = TestClient(app)
    default_a(client)
    resp = stitch(
        client,
        "scene-bad",
        [seg("scene-a", 1, 3000) for _ in range(count)],
    )
    assert resp.status_code == 422
    assert client.get("/scenes/scene-bad").status_code == 404


def test_stitch_missing_fields_422():
    client = TestClient(app)
    default_a(client)
    valid_seg = {"scene_id": "scene-a", "expected_revision": 1,
                 "duration_ms": 3000}
    for body in (
        {"segments": [valid_seg, valid_seg]},  # 缺 target_scene_id
        {"target_scene_id": "t"},              # 缺 segments
        {"target_scene_id": "t", "segments": [
            {"expected_revision": 1, "duration_ms": 3000},  # 缺 scene_id
            valid_seg,
        ]},
        {"target_scene_id": "t", "segments": [
            {"scene_id": "scene-a", "duration_ms": 3000},  # 缺 expected_revision
            valid_seg,
        ]},
        {"target_scene_id": "t", "segments": [
            {"scene_id": "scene-a", "expected_revision": 1},  # 缺 duration_ms
            valid_seg,
        ]},
        {"target_scene_id": "t", "segments": None},
        {"target_scene_id": None, "segments": [valid_seg, valid_seg]},
    ):
        resp = client.post("/stitch", json=body)
        assert resp.status_code == 422, body


@pytest.mark.parametrize("field", ["expected_revision", "duration_ms"])
@pytest.mark.parametrize("value", [True, False, "1", "3000", 1.0, 0, -1, 2.5])
def test_stitch_integer_fields_reject_bools_strings_floats_nonpositive(
    field, value
):
    client = TestClient(app)
    default_a(client)
    bad = {"scene_id": "scene-a", "expected_revision": 1, "duration_ms": 3000}
    bad[field] = value
    resp = stitch(client, "scene-bad", [bad, {
        "scene_id": "scene-a", "expected_revision": 1, "duration_ms": 3000,
    }])
    assert resp.status_code == 422, resp.text
    assert client.get("/scenes/scene-bad").status_code == 404


@pytest.mark.parametrize("target", ["", "   "])
def test_stitch_blank_target_scene_id_422(target):
    client = TestClient(app)
    default_a(client)
    resp = stitch(
        client, target,
        [seg("scene-a", 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("scene_id", ["", "   "])
def test_stitch_blank_segment_scene_id_422(scene_id):
    client = TestClient(app)
    default_a(client)
    resp = stitch(
        client, "scene-bad",
        [seg(scene_id, 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 422
    assert client.get("/scenes/scene-bad").status_code == 404


def test_stitch_non_string_scene_ids_422():
    client = TestClient(app)
    default_a(client)
    resp = stitch(
        client, "scene-bad",
        [seg(123, 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 422


def test_stitch_mismatched_track_counts_422():
    client = TestClient(app)
    default_a(client)                       # 2 轨
    default_b(client, "scene-b", track_count=3)  # 3 轨

    resp = stitch(
        client, "scene-bad",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 2000)],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["segment"] == 2
    assert detail["scene_id"] == "scene-b"
    assert detail["track_count"] == 3
    assert detail["expected_track_count"] == 2

    # 第一段就是异轨数（以第一段为准）
    resp = stitch(
        client, "scene-bad2",
        [seg("scene-b", 1, 2000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["segment"] == 2

    assert client.get("/scenes/scene-bad").status_code == 404
    assert client.get("/scenes/scene-bad2").status_code == 404


def test_stitch_duration_shorter_than_source_max_end_422():
    client = TestClient(app)
    default_a(client)  # max end_ms = 3000
    default_b(client)  # max end_ms = 2000

    resp = stitch(
        client, "scene-bad",
        [seg("scene-a", 1, 2999), seg("scene-b", 1, 2000)],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["segment"] == 1
    assert detail["scene_id"] == "scene-a"
    assert detail["duration_ms"] == 2999
    assert detail["max_end_ms"] == 3000

    # 第二段过短
    resp = stitch(
        client, "scene-bad2",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 1999)],
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["segment"] == 2
    assert client.get("/scenes/scene-bad").status_code == 404
    assert client.get("/scenes/scene-bad2").status_code == 404


def test_stitch_total_subtitles_over_12_returns_422():
    client = TestClient(app)
    # 7 条互不重叠字幕（2 轨即可）+ 6 条 = 13
    big = [
        {"id": f"x{i}", "speaker": f"S{i}", "start_ms": i * 10,
         "end_ms": i * 10 + 5}
        for i in range(7)
    ]
    make_scene(client, "scene-big", big)
    six = [
        {"id": f"a{i}", "speaker": "A", "start_ms": i * 100,
         "end_ms": i * 100 + 50}
        for i in range(6)
    ]
    make_scene(client, "scene-six", six)

    resp = stitch(
        client, "scene-over",
        [seg("scene-six", 1, 550), seg("scene-big", 1, 65)],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["subtitle_count"] == 13
    assert client.get("/scenes/scene-over").status_code == 404


# ---------------------------------------------------------------------------
# 404 / 409
# ---------------------------------------------------------------------------


def test_stitch_unknown_source_returns_404():
    client = TestClient(app)
    default_a(client)

    resp = stitch(
        client, "scene-new",
        [seg("no-such", 1, 100), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["segment"] == 1
    assert detail["scene_id"] == "no-such"
    assert client.get("/scenes/scene-new").status_code == 404

    # 第二段不存在同样 404（第一段有效）
    resp = stitch(
        client, "scene-new2",
        [seg("scene-a", 1, 3000), seg("ghost", 1, 100)],
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["segment"] == 2
    assert client.get("/scenes/scene-new2").status_code == 404


def test_stitch_stale_revision_returns_409():
    client = TestClient(app)
    default_a(client, "scene-rev")  # revision 1
    make_scene(
        client, "scene-rev",
        [{"id": "only", "speaker": "Solo", "start_ms": 0, "end_ms": 50}],
        expected_revision=1,
    )  # revision 2

    resp = stitch(
        client, "scene-out",
        [seg("scene-rev", 1, 50), seg("scene-rev", 2, 50)],
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["segment"] == 1
    assert detail["scene_id"] == "scene-rev"
    assert detail["current_revision"] == 2
    assert client.get("/scenes/scene-out").status_code == 404


def test_stitch_future_revision_returns_409():
    client = TestClient(app)
    default_a(client)
    resp = stitch(
        client, "scene-out",
        [seg("scene-a", 3, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["current_revision"] == 1
    assert client.get("/scenes/scene-out").status_code == 404


def test_stitch_target_already_exists_returns_409():
    client = TestClient(app)
    a = default_a(client, "scene-a")
    default_b(client, "scene-b")

    resp = stitch(
        client, "scene-b",
        [seg("scene-a", 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["scene_id"] == "scene-b"
    assert detail["current_revision"] == 1

    # 已存在目标保持原样；源不变
    got_b = client.get("/scenes/scene-b").json()
    assert [s["id"] for s in got_b["subtitles"]] == ["b1", "b2"]
    assert client.get("/scenes/scene-a").json() == a


def test_stitch_target_same_as_source_returns_409():
    client = TestClient(app)
    default_a(client, "scene-a")
    resp = stitch(
        client, "scene-a",
        [seg("scene-a", 1, 3000), seg("scene-a", 1, 3000)],
    )
    assert resp.status_code == 409
    src = client.get("/scenes/scene-a").json()
    assert src["revision"] == 1
    assert [s["id"] for s in src["subtitles"]] == ["a1", "a2", "a3"]


# ---------------------------------------------------------------------------
# 原子性、源/会话不变、目标可用
# ---------------------------------------------------------------------------


def test_stitch_does_not_modify_sources_or_playback_sessions():
    client = TestClient(app)
    a_before = default_a(client, "scene-keep")
    b_before = default_b(client, "scene-keep-b")
    sess = client.post(
        "/sessions", json={"scene_id": "scene-keep"}
    ).json()
    client.post(
        f"/sessions/{sess['session_id']}/advance", json={"time_ms": 900}
    )

    resp = stitch(
        client, "scene-keep-out",
        [seg("scene-keep", 1, 3000), seg("scene-keep-b", 1, 2000)],
    )
    assert resp.status_code == 200

    # 源快照逐字段不变
    assert client.get("/scenes/scene-keep").json() == a_before
    assert client.get("/scenes/scene-keep-b").json() == b_before

    # 既有播放会话不受拼接影响
    state = client.post(
        f"/sessions/{sess['session_id']}/advance", json={"time_ms": 900}
    ).json()
    assert state["cursor_ms"] == 900
    assert state["source_revision"] == 1


def test_stitch_target_supports_read_revise_and_playback():
    """拼接目标是普通场景：可读取、修订（revision 1 -> 2）、播放。"""
    client = TestClient(app)
    default_a(client)
    default_b(client)
    resp = stitch(
        client, "scene-play",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 2000)],
    )
    assert resp.status_code == 200

    # 读取
    assert client.get("/scenes/scene-play").json() == resp.json()

    # 修订
    rev = client.put(
        "/scenes/scene-play",
        json={
            "track_count": 2,
            "expected_revision": 1,
            "subtitles": [
                {"id": "only", "speaker": "Solo",
                 "start_ms": 0, "end_ms": 100},
            ],
        },
    )
    assert rev.status_code == 200
    assert rev.json()["revision"] == 2

    # 播放修订后的目标
    sess = client.post(
        "/sessions", json={"scene_id": "scene-play"}
    ).json()
    assert sess["source_revision"] == 2
    adv = client.post(
        f"/sessions/{sess['session_id']}/advance", json={"time_ms": 50}
    ).json()
    assert [a["id"] for a in adv["active"]] == ["only"]

    # 事件推进可用
    ev = client.post(
        f"/sessions/{sess['session_id']}/events", json={"time_ms": 100}
    ).json()
    assert ev["events"] == [
        {"id": "only", "speaker": "Solo", "track": 1,
         "time_ms": 100, "type": "exit"}
    ]


def test_stitch_target_plays_as_continuous_timeline():
    """播放拼接结果：事件时刻落在平移后的连续时间轴上。"""
    client = TestClient(app)
    default_a(client)
    default_b(client)
    stitch(
        client, "scene-cont",
        [seg("scene-a", 1, 3000), seg("scene-b", 1, 2000)],
    )
    sess = client.post(
        "/sessions", json={"scene_id": "scene-cont"}
    ).json()
    ev = client.post(
        f"/sessions/{sess['session_id']}/events", json={"time_ms": 5000}
    ).json()
    assert ev["events"] == [
        {"id": "1:2", "speaker": "Bob", "track": 2,
         "time_ms": 800, "type": "enter"},
        {"id": "1:1", "speaker": "Alice", "track": 1,
         "time_ms": 1000, "type": "exit"},
        {"id": "1:3", "speaker": "Alice", "track": 1,
         "time_ms": 1000, "type": "enter"},
        {"id": "1:3", "speaker": "Alice", "track": 1,
         "time_ms": 2500, "type": "exit"},
        {"id": "1:2", "speaker": "Bob", "track": 2,
         "time_ms": 3000, "type": "exit"},
        {"id": "2:1", "speaker": "Carol", "track": 1,
         "time_ms": 3000, "type": "enter"},
        {"id": "2:2", "speaker": "Alice", "track": 2,
         "time_ms": 3000, "type": "enter"},
        {"id": "2:1", "speaker": "Carol", "track": 1,
         "time_ms": 3500, "type": "exit"},
        {"id": "2:2", "speaker": "Alice", "track": 2,
         "time_ms": 5000, "type": "exit"},
    ]
    assert ev["active"] == []


def test_stitch_failure_leaves_no_target_and_sources_untouched():
    client = TestClient(app)
    a = default_a(client, "scene-fail-a")
    b = default_b(client, "scene-fail-b")

    # 轨道数不一致：422，无目标残留
    make_scene(
        client, "scene-three",
        [{"id": "t", "speaker": "T", "start_ms": 0, "end_ms": 10}],
        track_count=3,
    )
    resp = stitch(
        client, "scene-fail-out",
        [seg("scene-fail-a", 1, 3000), seg("scene-three", 1, 10)],
    )
    assert resp.status_code == 422
    assert client.get("/scenes/scene-fail-out").status_code == 404

    # 源不变
    assert client.get("/scenes/scene-fail-a").json() == a
    assert client.get("/scenes/scene-fail-b").json() == b


# ---------------------------------------------------------------------------
# 并发：拼接与源修订 / 并并发拼接
# ---------------------------------------------------------------------------


def test_concurrent_stitch_and_revision_never_mixes_snapshots():
    """并发拼接与源修订：写锁串行，成功则内容与某完整版本一致。"""
    client = TestClient(app)
    default_a(client, "scene-race")

    results = []
    barrier = threading.Barrier(2)

    def do_stitch():
        local = TestClient(app)
        barrier.wait()
        resp = stitch(
            local, "scene-race-out",
            [seg("scene-race", 1, 3000), seg("scene-race", 1, 3000)],
        )
        results.append(("stitch", resp.status_code, resp.json()))

    def do_revise():
        local = TestClient(app)
        barrier.wait()
        resp = local.put(
            "/scenes/scene-race",
            json={
                "track_count": 2,
                "expected_revision": 1,
                "subtitles": [
                    {"id": "only", "speaker": "Solo",
                     "start_ms": 0, "end_ms": 50},
                ],
            },
        )
        results.append(("revise", resp.status_code, resp.json()))

    threads = [threading.Thread(target=do_stitch),
               threading.Thread(target=do_revise)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stitch_result = next(r for r in results if r[0] == "stitch")
    revise_result = next(r for r in results if r[0] == "revise")
    assert revise_result[1] == 200
    assert revise_result[2]["revision"] == 2

    if stitch_result[1] == 200:
        body = stitch_result[2]
        assert body["revision"] == 1
        # 拼接先生效：两段都取自 revision 1 的三条字幕
        assert [s["id"] for s in body["subtitles"]] == [
            "1:1", "1:2", "1:3", "2:1", "2:2", "2:3",
        ]
    else:
        assert stitch_result[1] == 409
        assert stitch_result[2]["detail"]["current_revision"] == 2
        assert client.get("/scenes/scene-race-out").status_code == 404


def test_concurrent_stitches_to_same_target_only_one_succeeds():
    client = TestClient(app)
    default_a(client, "scene-multi")
    results = []
    barrier = threading.Barrier(4)

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = stitch(
            local, "scene-multi-out",
            [seg("scene-multi", 1, 3000), seg("scene-multi", 1, 3000)],
        )
        results.append(resp.status_code)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 1
    assert results.count(409) == 3
    assert client.get("/scenes/scene-multi-out").json()["revision"] == 1
    assert client.get("/scenes/scene-multi").json()["revision"] == 1


# ---------------------------------------------------------------------------
# 回归：源快照与 expected_revision 错配（可控交错）
# ---------------------------------------------------------------------------


def _patch_stitch_gates(monkeypatch, *, pause_at_read, pause_at_commit):
    """故障注入：在拼接路由的读取与提交边界受控暂停（仅第一次）。"""
    release_read = threading.Event()
    release_commit = threading.Event()
    store = main_module.get_store()
    real_get_scene = store.get_scene
    real_commit_stitch = store.commit_stitch
    read_done = []

    def wrapped_get_scene(scene_id):
        result = real_get_scene(scene_id)
        if not read_done:
            read_done.append(scene_id)
            pause_at_read.set()
            assert release_read.wait(timeout=10), "测试未及时放行被暂停的读取"
            # 故意不重新读取：放行后仍交付手中（可能已过时）的快照
        return result

    commit_done = []

    def wrapped_commit_stitch(*args, **kwargs):
        if not commit_done:
            commit_done.append(args)
            pause_at_commit.set()
            assert release_commit.wait(timeout=10), "测试未及时放行被暂停的提交"
        return real_commit_stitch(*args, **kwargs)

    monkeypatch.setattr(store, "get_scene", wrapped_get_scene)
    monkeypatch.setattr(store, "commit_stitch", wrapped_commit_stitch)
    return release_read, release_commit


def _post_stitch_in_thread(segments, responses, key="stitch", target="scene-mm-out"):
    local = TestClient(app)
    responses[key] = stitch(local, target, segments)


def test_stitch_reads_wrong_revision_then_source_catches_up_returns_409(
    monkeypatch,
):
    """expected_revision=2 先读到源 revision 1；暂停期间源升级到 2。

    读取即核对：立即 409，不创建目标、不改源场景、不动播放会话。
    """
    client = TestClient(app)
    default_a(client, "scene-mm")
    sid = client.post(
        "/sessions", json={"scene_id": "scene-mm"}
    ).json()["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    pause_at_read = threading.Event()
    pause_at_commit = threading.Event()
    release_read, release_commit = _patch_stitch_gates(
        monkeypatch,
        pause_at_read=pause_at_read,
        pause_at_commit=pause_at_commit,
    )
    release_commit.set()  # 本用例不应到达提交点

    responses = {}
    t = threading.Thread(
        target=_post_stitch_in_thread,
        args=([
            seg("scene-mm", 2, 3000),
            seg("scene-mm", 2, 3000),
        ], responses),
    )
    t.start()
    assert pause_at_read.wait(timeout=10), "拼接线程未读到源"

    # 源从 revision 1 升级到 2
    rev = client.put(
        "/scenes/scene-mm",
        json={
            "track_count": 2,
            "expected_revision": 1,
            "subtitles": [
                {"id": "only", "speaker": "Solo",
                 "start_ms": 0, "end_ms": 50},
            ],
        },
    )
    assert rev.status_code == 200

    release_read.set()
    t.join(timeout=10)
    assert not t.is_alive()

    resp = responses["stitch"]
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["scene_id"] == "scene-mm"
    assert detail["segment"] == 1
    assert detail["current_revision"] == 1  # 如实报告读到的版本

    assert client.get("/scenes/scene-mm-out").status_code == 404
    source = client.get("/scenes/scene-mm").json()
    assert source["revision"] == 2

    # 播放会话不受影响
    state = client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 900}
    ).json()
    assert state["source_revision"] == 1
    assert state["cursor_ms"] == 900


def test_stitch_source_revised_after_matched_read_returns_409(monkeypatch):
    """读取时版本匹配（1==1）；提交前任一源被修订到 2。

    事务内第二道核对必须拦住：409，不留目标，源停留在 revision 2。
    """
    client = TestClient(app)
    default_a(client, "scene-bump")
    default_b(client, "scene-bump-b")

    pause_at_read = threading.Event()
    pause_at_commit = threading.Event()
    release_read, release_commit = _patch_stitch_gates(
        monkeypatch,
        pause_at_read=pause_at_read,
        pause_at_commit=pause_at_commit,
    )

    responses = {}
    t = threading.Thread(
        target=_post_stitch_in_thread,
        args=([
            seg("scene-bump", 1, 3000),
            seg("scene-bump-b", 1, 2000),
        ], responses),
    )
    t.start()
    # 两段读取都会经过包装（仅第一次暂停）
    assert pause_at_read.wait(timeout=10)
    release_read.set()
    assert pause_at_commit.wait(timeout=10)

    # 提交前：第二个源升级到 revision 2
    rev = client.put(
        "/scenes/scene-bump-b",
        json={
            "track_count": 2,
            "expected_revision": 1,
            "subtitles": [
                {"id": "only", "speaker": "Solo",
                 "start_ms": 0, "end_ms": 50},
            ],
        },
    )
    assert rev.status_code == 200

    release_commit.set()
    t.join(timeout=10)
    assert not t.is_alive()

    resp = responses["stitch"]
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["scene_id"] == "scene-bump-b"
    assert detail["segment"] == 2
    assert detail["current_revision"] == 2

    assert client.get("/scenes/scene-mm-out").status_code == 404
    assert client.get("/scenes/scene-bump").json()["revision"] == 1
    assert client.get("/scenes/scene-bump-b").json()["revision"] == 2


def test_stitch_uninterrupted_flow_through_gates_succeeds(monkeypatch):
    """两道核对之间源版本不变：正常拼接成功，目标可读取、修订、播放。"""
    client = TestClient(app)
    a_before = default_a(client, "scene-ok")
    b_before = default_b(client, "scene-ok-b")

    pause_at_read = threading.Event()
    pause_at_commit = threading.Event()
    release_read, release_commit = _patch_stitch_gates(
        monkeypatch,
        pause_at_read=pause_at_read,
        pause_at_commit=pause_at_commit,
    )

    responses = {}
    t = threading.Thread(
        target=_post_stitch_in_thread,
        args=([
            seg("scene-ok", 1, 3000),
            seg("scene-ok-b", 1, 2000),
        ], responses, "stitch", "scene-ok-out"),
    )
    t.start()
    assert pause_at_read.wait(timeout=10)
    release_read.set()
    assert pause_at_commit.wait(timeout=10)
    release_commit.set()
    t.join(timeout=10)
    assert not t.is_alive()

    resp = responses["stitch"]
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scene_id"] == "scene-ok-out"
    assert body["revision"] == 1
    assert [s["id"] for s in body["subtitles"]] == [
        "1:1", "1:2", "1:3", "2:1", "2:2",
    ]

    assert client.get("/scenes/scene-ok-out").json() == body
    assert client.get("/scenes/scene-ok").json() == a_before
    assert client.get("/scenes/scene-ok-b").json() == b_before

    # 目标可修订
    rev = client.put(
        "/scenes/scene-ok-out",
        json={
            "track_count": 2,
            "expected_revision": 1,
            "subtitles": [
                {"id": "z", "speaker": "Z", "start_ms": 0, "end_ms": 10},
            ],
        },
    )
    assert rev.status_code == 200
    assert rev.json()["revision"] == 2

    # 目标可播放
    sess = client.post(
        "/sessions", json={"scene_id": "scene-ok-out"}
    ).json()
    assert sess["source_revision"] == 2
