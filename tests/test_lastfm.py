"""
Unit tests for Last.fm API Client (Phase 10).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.lastfm.client import LastFMClient


class TestLastFMClientOffline:

    def test_offline_mode_detected(self):
        client = LastFMClient(api_key="YOUR_API_KEY_HERE")
        assert not client.is_live

    def test_empty_key_is_offline(self):
        client = LastFMClient(api_key="")
        assert not client.is_live

    def test_artist_info_offline_fallback(self):
        client = LastFMClient(api_key="")
        info = client.get_artist_info("radiohead")
        assert info["name"] == "radiohead"
        assert not info["is_live"]
        assert "radiohead" in info["url"].lower()
        assert isinstance(info["tags"], list)

    def test_top_tracks_offline_fallback(self):
        client = LastFMClient(api_key="")
        tracks = client.get_top_tracks("daft punk", limit=3)
        assert len(tracks) == 3
        assert all(t["artist"] == "daft punk" for t in tracks)
        assert all(not t["is_live"] for t in tracks)

    def test_similar_artists_offline_fallback(self):
        client = LastFMClient(api_key="")
        similar = client.get_similar_artists("nirvana")
        assert similar == []


class TestLastFMClientLiveMocked:

    @patch("requests.get")
    def test_artist_info_parsed_correctly(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artist": {
                "name": "Radiohead",
                "bio": {"summary": "Radiohead are an English rock band. <a href='...'>Read more</a>"},
                "tags": {"tag": [{"name": "alternative rock"}, {"name": "electronic"}]},
                "stats": {"listeners": "5000000", "playcount": "100000000"},
                "url": "https://www.last.fm/music/Radiohead",
                "image": [{"#text": "https://img.last.fm/radiohead_small.jpg"}, {"#text": "https://img.last.fm/radiohead_large.jpg"}],
            }
        }
        mock_get.return_value = mock_resp

        client = LastFMClient(api_key="valid_test_key_123")
        info = client.get_artist_info("radiohead")

        assert info["name"] == "Radiohead"
        assert info["is_live"]
        assert "Radiohead are an English rock band." in info["bio"]
        assert "<a href" not in info["bio"]
        assert info["tags"] == ["alternative rock", "electronic"]
        assert info["listeners"] == 5000000
        assert info["image_url"] == "https://img.last.fm/radiohead_large.jpg"

    @patch("requests.get")
    def test_caching_avoids_duplicate_network_calls(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "toptracks": {
                "track": [
                    {"name": "Karma Police", "listeners": "1000000", "playcount": "2000000", "url": "https://..."},
                ]
            }
        }
        mock_get.return_value = mock_resp

        client = LastFMClient(api_key="valid_test_key_123")
        # Call twice with same arguments
        tracks1 = client.get_top_tracks("radiohead", limit=1)
        tracks2 = client.get_top_tracks("radiohead", limit=1)

        assert tracks1 == tracks2
        # Network request should have happened only once due to cache
        assert mock_get.call_count == 1
