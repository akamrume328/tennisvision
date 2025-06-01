"""
テニスビジョンプロジェクト メインランチャー
全機能を統合した実行ファイル
"""

def main():
    """メインランチャー"""
    print("=== テニスビジョンプロジェクト ===")
    print()
    print("利用可能な機能:")
    print()
    print("1. 局面アノテーション（データ収集）")
    print("   - 動画の局面ラベリング")
    print("   - コート座標設定")
    print("   - 訓練用データ作成")
    print()
    print("2. 局面分類モデル訓練")
    print("   - アノテーションデータからモデル学習")
    print("   - LSTM局面分類器の訓練")
    print()
    print("3. ボール追跡")
    print("   - YOLO v8によるボール検出")
    print("   - リアルタイム追跡")
    print("   - 軌跡データ出力")
    print()
    print("4. 統合分析（ボール追跡 + 局面判断）")
    print("   - ボール追跡と局面判断の同時実行")
    print("   - リアルタイム結果表示")
    print()
    print("5. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-5): ").strip()
            
            if choice == '1':
                # 局面アノテーション
                print("\n🏷️  局面アノテーションツールを起動中...")
                try:
                    from tennis_ball_tracking.data_collector import main as data_collector_main
                    data_collector_main()
                except Exception as e:
                    print(f"❌ エラー: {e}")
                break
                
            elif choice == '2':
                # モデル訓練
                print("\n🤖 局面分類モデル訓練を開始中...")
                try:
                    from tennis_ball_tracking.train_phase_model_back import main as train_main
                    # train_phase_model.pyはif __name__ == "__main__":で直接実行される
                    exec(open('tennis_ball_tracking/train_phase_model.py').read())
                except Exception as e:
                    print(f"❌ エラー: {e}")
                break
                
            elif choice == '3':
                # ボール追跡
                print("\n🎾 ボール追跡ツールを起動中...")
                try:
                    from tennis_ball_tracking.balltracking import main as ball_tracking_main
                    ball_tracking_main()
                except Exception as e:
                    print(f"❌ エラー: {e}")
                break
                
            elif choice == '4':
                # 統合分析
                print("\n📊 統合分析ツールを起動中...")
                try:
                    from situation_analyze.main_demo import main as demo_main
                    demo_main()
                except Exception as e:
                    print(f"❌ エラー: {e}")
                break
                
            elif choice == '5':
                print("終了します")
                break
                
            else:
                print("無効な選択です。1-5を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n操作がキャンセルされました")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            break

if __name__ == "__main__":
    main()
