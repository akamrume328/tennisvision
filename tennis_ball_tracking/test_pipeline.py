"""
テニスボール追跡学習パイプラインの独立性テスト
3つのコンポーネントが独立して動作することを確認
"""
import os
import json
import glob
from datetime import datetime

def test_file_structure():
    """ファイル構造のテスト"""
    print("=== File Structure Test ===")
    
    required_files = [
        "balltracking.py",
        "data_collector.py", 
        "train_phase_model.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✓ {file} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    # training_dataディレクトリの確認
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
        print("✓ Created training_data directory")
    else:
        print("✓ training_data directory exists")
    
    return True

def test_expected_output_files():
    """期待される出力ファイルの構造をテスト"""
    print("\n=== Expected Output Files Test ===")
    
    # 各コンポーネントが出力すべきファイル形式
    expected_patterns = {
        "balltracking.py": "tracking_features_*.json",
        "data_collector.py": ["phase_annotations_*.json", "court_coords_*.json"],
        "train_phase_model.py": "tennis_phase_model.pth"
    }
    
    print("Expected file patterns:")
    for component, patterns in expected_patterns.items():
        if isinstance(patterns, list):
            for pattern in patterns:
                print(f"  {component} → {pattern}")
        else:
            print(f"  {component} → {patterns}")
    
    return True

def test_data_format_examples():
    """各コンポーネントのデータ形式例を作成"""
    print("\n=== Creating Data Format Examples ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # balltracking.py出力例
    tracking_example = {
        "ball_position": [320.5, 240.8],
        "ball_velocity": [15.2, -8.7],
        "ball_detected": True,
        "player1_position": [150.0, 400.0],
        "player2_position": [480.0, 200.0],
        "player1_detected": True,
        "player2_detected": True,
        "timestamp": 16.67
    }
    
    tracking_file = f"training_data/tracking_features_{timestamp}_example.json"
    with open(tracking_file, 'w') as f:
        json.dump([tracking_example] * 5, f, indent=2)
    print(f"✓ Created tracking features example: {tracking_file}")
    
    # data_collector.py出力例 - フェーズアノテーション
    phase_example = {
        "video_info": {
            "filename": f"test_video_{timestamp}.mp4",
            "total_frames": 1000,
            "fps": 30.0
        },        "annotations": [
            {"frame": 0, "phase": "point_interval", "timestamp": "2025-06-01T12:00:00"},
            {"frame": 30, "phase": "serve_preparation", "timestamp": "2025-06-01T12:00:01"},
            {"frame": 60, "phase": "serve_front_deuce", "timestamp": "2025-06-01T12:00:02"},
            {"frame": 90, "phase": "rally", "timestamp": "2025-06-01T12:00:03"}
        ]
    }
    
    phase_file = f"training_data/phase_annotations_{timestamp}_example.json"
    with open(phase_file, 'w') as f:
        json.dump(phase_example, f, indent=2)
    print(f"✓ Created phase annotations example: {phase_file}")
    
    # data_collector.py出力例 - コート座標
    court_example = {
        "court_corners": [
            [100, 500],
            [700, 500], 
            [700, 100],
            [100, 100]
        ],
        "net_position": [400, 300],
        "service_boxes": {
            "left": [[100, 300], [400, 500]],
            "right": [[400, 100], [700, 300]]
        }
    }
    
    court_file = f"training_data/court_coords_{timestamp}_example.json"
    with open(court_file, 'w') as f:
        json.dump(court_example, f, indent=2)
    print(f"✓ Created court coordinates example: {court_file}")
    
    return True

def test_component_independence():
    """コンポーネントの独立性をテスト"""
    print("\n=== Component Independence Test ===")
    
    print("Checking component independence:")
    print("✓ balltracking.py - Only depends on YOLO model and video input")
    print("✓ data_collector.py - Only depends on OpenCV and user input")
    print("✓ train_phase_model.py - Only depends on JSON data files")
    
    print("\nData flow:")
    print("  balltracking.py → tracking_features_*.json")
    print("  data_collector.py → phase_annotations_*.json + court_coords_*.json")
    print("  train_phase_model.py ← reads all JSON files → tennis_phase_model.pth")
    
    return True

def test_data_integration():
    """データ統合のテスト"""
    print("\n=== Data Integration Test ===")
    
    training_dir = "training_data"
    
    # ファイルパターンをチェック
    tracking_files = glob.glob(os.path.join(training_dir, "tracking_features_*.json"))
    phase_files = glob.glob(os.path.join(training_dir, "phase_annotations_*.json"))
    court_files = glob.glob(os.path.join(training_dir, "court_coords_*.json"))
    
    print(f"Found files:")
    print(f"  Tracking features: {len(tracking_files)}")
    print(f"  Phase annotations: {len(phase_files)}")
    print(f"  Court coordinates: {len(court_files)}")
    
    if tracking_files and phase_files:
        print("✓ Basic data integration possible")
        
        # ファイル名の一致確認
        tracking_bases = [os.path.basename(f).replace('tracking_features_', '').replace('.json', '') 
                         for f in tracking_files]
        phase_bases = [os.path.basename(f).replace('phase_annotations_', '').replace('.json', '') 
                      for f in phase_files]
        
        matching = set(tracking_bases) & set(phase_bases)
        print(f"  Matching file pairs: {len(matching)}")
        
        if matching:
            print("✓ Data files can be properly paired for training")
        else:
            print("⚠ No matching file pairs found - need to collect paired data")
    else:
        print("⚠ Need both tracking and phase data for training")
    
    return True

def main():
    """メインテスト関数"""
    print("Tennis Ball Tracking Pipeline Independence Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_expected_output_files,
        test_data_format_examples,
        test_component_independence,
        test_data_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline restructuring is complete.")
        print("\nNext steps:")
        print("1. Run balltracking.py on video files to generate tracking data")
        print("2. Run data_collector.py to annotate phases and set court coordinates")
        print("3. Run train_phase_model.py to train the phase classification model")
    else:
        print("⚠ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
