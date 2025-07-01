import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any # <- Dict, Any を追加
import re # 正規表現モジュールをインポート
import json # JSONモジュールをインポート

class MergeFeaturesAndPredictions:
    """
    教師データCSV (feature_extractor.py出力) と LSTM予測CSV (predict_lstm_model.py出力) をマージするクラス。
    マージ時に、HMM学習用のメタデータも生成・保存します。
    """
    def __init__(self, 
                 output_dir_name: str = "merged_predictions",
                 base_dir: Path = Path("./training_data")):
        """
        MergeFeaturesAndPredictions を初期化します。

        Args:
            output_dir_name (str): マージされたCSVファイルの保存先ディレクトリ名。
            base_dir (Path): `training_data` などの基準ディレクトリ。
        """
        self.base_dir = base_dir
        self.output_dir = self.base_dir / output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"マージ結果の保存先: {self.output_dir.resolve()}")
        print(f"HMM用メタデータも上記ディレクトリに保存されます。")

    def _select_file_dialog(self, title: str, directory: Path, pattern: str) -> Optional[Path]:
        """
        簡易的なファイル選択ダイアログ（コンソール）。
        指定されたディレクトリとパターンに一致するファイルを表示し、ユーザーに選択を促します。
        """
        print(f"\n=== {title} ===")
        if not directory.exists():
            print(f"❌ 指定されたディレクトリが見つかりません: {directory}")
            return None
            
        files = sorted(list(directory.glob(pattern)))
        if not files:
            print(f"❌ {pattern} に一致するファイルが見つかりません in {directory}")
            return None

        for i, f_path in enumerate(files, 1):
            try:
                mtime = datetime.fromtimestamp(f_path.stat().st_mtime)
                print(f"  {i}. {f_path.name} (更新日時: {mtime:%Y-%m-%d %H:%M})")
            except FileNotFoundError:
                print(f"  {i}. {f_path.name} (ファイルが見つかりません。スキップします。)")
                continue # ファイルが存在しない場合はスキップ
        
        if not any(f.exists() for f in files): # 有効なファイルが一つもなければ終了
            print(f"❌ 有効なファイルが一つも見つかりませんでした。")
            return None

        while True:
            try:
                choice_str = input(f"選択してください (1-{len(files)}), 0でスキップ: ").strip()
                choice_num = int(choice_str)
                if choice_num == 0:
                    return None
                if 1 <= choice_num <= len(files):
                    selected_file = files[choice_num - 1]
                    if selected_file.exists():
                        return selected_file
                    else:
                        print(f"選択されたファイル {selected_file.name} は存在しません。再度選択してください。")
                else:
                    print(f"無効な選択です。1から{len(files)}の間で入力してください。")
            except ValueError:
                print("無効な入力です。数値を入力してください。")
            except IndexError:
                print("選択肢の範囲外です。")


    def load_csv(self, file_path: Path, file_description: str) -> Optional[pd.DataFrame]:
        """
        CSVファイルを読み込みます。

        Args:
            file_path (Path): 読み込むCSVファイルのパス。
            file_description (str): ファイルの説明（例: "教師データCSV"）。

        Returns:
            Optional[pd.DataFrame]: 読み込まれたDataFrame。エラー時はNone。
        """
        if not file_path or not file_path.exists():
            print(f"❌ {file_description}のパスが無効またはファイルが存在しません: {file_path}")
            return None
        try:
            df = pd.read_csv(file_path)
            print(f"✅ {file_description}を読み込みました: {file_path.name} ({len(df)}行)")
            return df
        except Exception as e:
            print(f"❌ {file_description}の読み込み中にエラーが発生しました ({file_path.name}): {e}")
            return None

    def merge_data(self, 
                   teacher_df: pd.DataFrame, 
                   prediction_df: pd.DataFrame,
                   teacher_true_label_col: str,
                   prediction_pred_label_col: str,
                   merge_keys: Optional[List[str]] = None
                   ) -> Optional[pd.DataFrame]:
        """
        教師データDataFrameと予測データDataFrameをマージします。

        Args:
            teacher_df (pd.DataFrame): 教師データ（真のラベルを含む）。
            prediction_df (pd.DataFrame): LSTM予測データ。
            teacher_true_label_col (str): 教師データDF内の真のラベル列名。
            prediction_pred_label_col (str): 予測データDF内の予測ラベル列名。
            merge_keys (Optional[List[str]]): マージに使用するキー列名のリスト。
                                            デフォルトは ['video_name', 'frame_number']。

        Returns:
            Optional[pd.DataFrame]: マージされたDataFrame。エラー時はNone。
        """
        if merge_keys is None:
            merge_keys = ['video_name', 'frame_number']

        print(f"\n--- データマージ開始 ---")
        print(f"マージキー: {merge_keys}")
        print(f"教師データ 真ラベル列: {teacher_true_label_col}")
        print(f"予測データ 予測ラベル列: {prediction_pred_label_col}")

        # マージキーの存在チェックとデータ型調整
        for key in merge_keys:
            if key not in teacher_df.columns:
                print(f"❌ 教師データにマージキー '{key}' が存在しません。")
                return None
            if key not in prediction_df.columns:
                print(f"❌ 予測データにマージキー '{key}' が存在しません。")
                return None
            
            # データ型を統一する試み
            try:
                if 'frame_number' in key.lower() or 'index' in key.lower(): # フレーム番号やインデックスらしき列は整数型に
                    print(f"ℹ️ マージキー '{key}' を整数型に変換します。")
                    teacher_df[key] = pd.to_numeric(teacher_df[key], errors='coerce').astype('Int64') # Nullable Integer
                    prediction_df[key] = pd.to_numeric(prediction_df[key], errors='coerce').astype('Int64')
                elif 'name' in key.lower() or 'video' in key.lower(): # 名前やビデオ名らしき列は文字列型に
                    print(f"ℹ️ マージキー '{key}' を文字列型に変換し、前後の空白を除去します。")
                    teacher_df[key] = teacher_df[key].astype(str).str.strip()
                    prediction_df[key] = prediction_df[key].astype(str).str.strip()
                    
                    # 予測データの video_name から日付サフィックス (_YYYYMMDD) を除去
                    if key == 'video_name':
                        print(f"ℹ️ 予測データの '{key}' から日付サフィックスを除去します。")
                        # 例: "video1_20230101" -> "video1"
                        prediction_df[key] = prediction_df[key].apply(lambda x: re.sub(r'_\d{8}$', '', x))
                
                # 変換後にNaNが多数発生していないか確認（デバッグ用）
                if teacher_df[key].isnull().sum() > 0.5 * len(teacher_df): # 半数以上NaNなら警告
                    print(f"⚠️ 教師データのキー '{key}' に変換後NaNが多数あります。元のデータを確認してください。")
                if prediction_df[key].isnull().sum() > 0.5 * len(prediction_df):
                    print(f"⚠️ 予測データのキー '{key}' に変換後NaNが多数あります。元のデータを確認してください。")

            except Exception as e:
                print(f"❌ マージキー '{key}' のデータ型変換中にエラー: {e}")
                return None

        if teacher_true_label_col not in teacher_df.columns:
            print(f"❌ 教師データに真のラベル列 '{teacher_true_label_col}' が存在しません。")
            return None
        if prediction_pred_label_col not in prediction_df.columns:
            print(f"❌ 予測データに予測ラベル列 '{prediction_pred_label_col}' が存在しません。")
            return None

        # デバッグ情報: マージキーの最初の数行を表示
        print("\n--- マージキーのサンプルデータ (変換後) ---")
        for key in merge_keys:
            print(f"教師データ '{key}' (型: {teacher_df[key].dtype}):\n{teacher_df[key].head()}")
            print(f"予測データ '{key}' (型: {prediction_df[key].dtype}):\n{prediction_df[key].head()}")
            # ユニークな値の数も再確認
            print(f"  教師データ '{key}' ユニーク数: {teacher_df[key].nunique()}")
            print(f"  予測データ '{key}' ユニーク数: {prediction_df[key].nunique()}")
            # 共通のキーの値があるか確認 (最初のいくつか)
            common_keys_sample = pd.Series(list(set(teacher_df[key].dropna().unique()) & set(prediction_df[key].dropna().unique()))).head()
            if not common_keys_sample.empty:
                print(f"  '{key}' の共通の値 (サンプル): {common_keys_sample.tolist()}")
            else:
                print(f"  '{key}' に共通の値が見つかりません。")
        
        # video_name の詳細比較
        if 'video_name' in merge_keys:
            print("\n--- 詳細デバッグ: video_name の比較 ---")
            teacher_video_names = set(teacher_df['video_name'].astype(str).unique())
            prediction_video_names = set(prediction_df['video_name'].astype(str).unique())

            common_video_names = sorted(list(teacher_video_names.intersection(prediction_video_names)))
            teacher_only_video_names = sorted(list(teacher_video_names.difference(prediction_video_names)))
            prediction_only_video_names = sorted(list(prediction_video_names.difference(teacher_video_names)))

            print(f"教師データにのみ存在するvideo_name ({len(teacher_only_video_names)}件): {teacher_only_video_names[:10]}") # 先頭10件表示
            if len(teacher_only_video_names) > 10: print("  ...")
            print(f"予測データにのみ存在するvideo_name ({len(prediction_only_video_names)}件): {prediction_only_video_names[:10]}")
            if len(prediction_only_video_names) > 10: print("  ...")
            print(f"両方のデータに存在するvideo_name ({len(common_video_names)}件): {common_video_names[:10]}")
            if len(common_video_names) > 10: print("  ...")

            if not common_video_names:
                print("❌ 共通のvideo_nameが見つかりません。video_nameの命名規則や値を確認してください。")
                # return None # ここで処理を中断する場合はコメントアウトを外す
            elif common_video_names and 'frame_number' in merge_keys:
                print("\n--- 詳細デバッグ: 共通video_nameにおけるframe_numberの比較 (最初の共通video_nameについて) ---")
                sample_video_name = common_video_names[0]
                
                teacher_sample_frames_df = teacher_df[teacher_df['video_name'] == sample_video_name]
                prediction_sample_frames_df = prediction_df[prediction_df['video_name'] == sample_video_name]

                if teacher_sample_frames_df.empty:
                    print(f"  ビデオ '{sample_video_name}': 教師データにフレームがありません。")
                elif prediction_sample_frames_df.empty:
                     print(f"  ビデオ '{sample_video_name}': 予測データにフレームがありません。")
                else:
                    teacher_sample_frames = teacher_sample_frames_df['frame_number']
                    prediction_sample_frames = prediction_sample_frames_df['frame_number']
                    print(f"ビデオ: {sample_video_name}")
                    print(f"  教師データ frame_number (min/max/nunique/type/nulls/sample): {teacher_sample_frames.min()}/{teacher_sample_frames.max()}/{teacher_sample_frames.nunique()}/{teacher_sample_frames.dtype}/{teacher_sample_frames.isnull().sum()}/{teacher_sample_frames.dropna().head().tolist()}")
                    print(f"  予測データ frame_number (min/max/nunique/type/nulls/sample): {prediction_sample_frames.min()}/{prediction_sample_frames.max()}/{prediction_sample_frames.nunique()}/{prediction_sample_frames.dtype}/{prediction_sample_frames.isnull().sum()}/{prediction_sample_frames.dropna().head().tolist()}")
                    
                    # 共通のフレーム番号
                    common_frames = set(teacher_sample_frames.dropna().unique()) & set(prediction_sample_frames.dropna().unique())
                    print(f"  共通のframe_number ({len(common_frames)}件): {sorted(list(common_frames))[:10]}")
                    if len(common_frames) > 10: print("    ...")


        try:
            # 予測DFからマージキーと予測ラベル列のみを選択（重複列を避けるため）
            # ただし、他の有用な列（例：信頼度）がある場合は、それも残すことを検討
            cols_to_keep_from_prediction = merge_keys + [prediction_pred_label_col]
            # prediction_df に他の有用な列があればここに追加
            # 例: if 'confidence' in prediction_df.columns: cols_to_keep_from_prediction.append('confidence')
            
            # 存在しない列を参照しようとするとエラーになるため、存在する列のみを選択
            cols_to_keep_from_prediction = [col for col in cols_to_keep_from_prediction if col in prediction_df.columns]
            # マージキーが cols_to_keep_from_prediction に含まれていることを確認
            for key_col in merge_keys:
                if key_col not in cols_to_keep_from_prediction:
                    cols_to_keep_from_prediction.append(key_col)
            cols_to_keep_from_prediction = list(dict.fromkeys(cols_to_keep_from_prediction)) # 重複削除しつつ順序維持

            prediction_subset_df = prediction_df[cols_to_keep_from_prediction].copy()

            # マージ実行 (inner join を使用し、両方のDFに存在するデータのみを保持)
            # HMMの後処理では、真のラベルと予測ラベルの両方が必要になるため。
            merged_df = pd.merge(teacher_df, prediction_subset_df, on=merge_keys, how='inner')
            
            print(f"✅ マージ成功: {len(merged_df)}行のデータがマージされました。")
            if len(merged_df) == 0:
                print(f"⚠️  マージ後のデータが0行です。マージキーが一致するデータがなかった可能性があります。")
                print(f"  教師データのキーのユニーク数: {[teacher_df[key].nunique() for key in merge_keys]}")
                print(f"  予測データのキーのユニーク数: {[prediction_df[key].nunique() for key in merge_keys]}")


            # 必要に応じて列名を標準化 (例: HMMが期待する列名に)
            # この例では、hmm_postprocessor.py のデフォルト引数に合わせておく
            # merged_df.rename(columns={teacher_true_label_col: 'true_phase', 
            #                           prediction_pred_label_col: 'predicted_phase'}, 
            #                  inplace=True)
            # print(f"ℹ️ 列名をリネーム: '{teacher_true_label_col}' -> 'true_phase', '{prediction_pred_label_col}' -> 'predicted_phase'")


            return merged_df
        except Exception as e:
            print(f"❌ データマージ中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _derive_labels_from_df(self, df: pd.DataFrame, column: str) -> Tuple[Optional[List[str]], Optional[Dict[str, int]], Optional[Dict[int, str]]]:
        """DataFrameの指定された単一列からユニークなラベルを抽出し、マッピングを作成する"""
        if column not in df.columns:
            print(f"⚠️警告: 指定された列 '{column}' がDataFrameに存在しません。ラベルを派生できません。")
            return None, None, None
        
        # NaNや空文字列を除外
        unique_labels = {label for label in df[column].unique() if pd.notna(label) and str(label).strip() != ""}
        if not unique_labels:
            print(f"⚠️警告: 列 '{column}' から有効なラベルが見つかりませんでした。")
            return None, None, None
            
        sorted_labels = sorted(list(unique_labels))
        label_to_int_map = {label: i for i, label in enumerate(sorted_labels)}
        int_to_label_map = {i: label for i, label in enumerate(sorted_labels)}
        print(f"ℹ️ DataFrameの列 '{column}' からラベル情報を派生: {label_to_int_map}")
        return sorted_labels, label_to_int_map, int_to_label_map

    def _load_teacher_metadata_labels(self, teacher_csv_path: Path) -> Tuple[Optional[List[str]], Optional[Dict[str, int]], Optional[Dict[int, str]]]:
        """教師データCSVに対応するメタデータJSONからラベル情報を読み込む"""
        if not teacher_csv_path:
            return None, None, None
            
        # メタデータファイル名の推測 (例: my_features.csv -> my_features_metadata.json)
        metadata_filename = teacher_csv_path.stem + "_metadata.json"
        # 教師データCSVと同じディレクトリにあると仮定
        potential_metadata_path = teacher_csv_path.parent / metadata_filename
        
        # LSTMモデルのメタデータも探す (より一般的な 'phase_labels' を持つ可能性)
        # 例: training_data/lstm_models/some_model_set/tennis_pytorch_metadata_*.json
        # これはより複雑な探索ロジックが必要になるため、今回は教師データ直結のメタデータのみを優先
        
        if potential_metadata_path.exists():
            print(f"ℹ️ 教師データ用メタデータファイルを検出: {potential_metadata_path}")
            try:
                with open(potential_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 'phase_labels' または 'labels' キーを探す
                phase_labels = metadata.get('phase_labels') or metadata.get('labels')
                
                if phase_labels and isinstance(phase_labels, list):
                    sorted_labels = sorted(list(set(phase_labels))) # 重複除去とソート
                    label_to_int_map = {label: i for i, label in enumerate(sorted_labels)}
                    int_to_label_map = {i: label for i, label in enumerate(sorted_labels)}
                    print(f"✅ 教師メタデータからラベル情報をロード: {label_to_int_map}")
                    return sorted_labels, label_to_int_map, int_to_label_map
                else:
                    print(f"⚠️警告: 教師メタデータに有効な 'phase_labels' または 'labels' (リスト形式) が見つかりません。")
            except Exception as e:
                print(f"❌ 教師メタデータの読み込み/解析エラー ({potential_metadata_path.name}): {e}")
        else:
            print(f"ℹ️情報: 教師データCSV ({teacher_csv_path.name}) に対応するメタデータファイル ({metadata_filename}) が見つかりませんでした。")
            
        return None, None, None

    def save_hmm_metadata(self, 
                          label_info: Tuple[List[str], Dict[str, int], Dict[int, str]], 
                          merged_csv_path: Path) -> Optional[Path]:
        """HMM学習用のメタデータをJSONファイルとして保存する"""
        sorted_labels, label_to_int, int_to_label = label_info
        
        if not sorted_labels or not label_to_int or not int_to_label:
            print("❌ HMMメタデータ保存エラー: ラベル情報が不完全です。")
            return None

        n_components = len(sorted_labels)
        if n_components == 0:
            print("❌ HMMメタデータ保存エラー: ラベル数が0です。")
            return None

        print(f"\n--- HMM用メタデータ保存 ---")
        try:
            metadata_content = {
                "n_components": int(n_components), 
                "phase_labels": [str(lbl) for lbl in sorted_labels], # 各ラベルを文字列に変換
                "label_to_int": {str(k): int(v) for k, v in label_to_int.items()}, # キーを文字列に、値をintに変換
                "int_to_label": {str(int(k)): str(v) for k, v in int_to_label.items()}, # キーをintにしてから文字列に、値を文字列に変換
                "source_merged_csv": merged_csv_path.name,
                "creation_timestamp": datetime.now().isoformat()
            }
            
            # マージ済みCSV名に基づいてHMMメタデータファイル名を生成
            # 例: merged_X_vs_Y_timestamp.csv -> hmm_metadata_merged_X_vs_Y_timestamp.json
            hmm_metadata_filename = f"hmm_metadata_{merged_csv_path.stem}.json"
            output_path = merged_csv_path.parent / hmm_metadata_filename # マージ済みCSVと同じディレクトリ

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_content, f, indent=4, ensure_ascii=False)
            
            print(f"✅ HMM用メタデータを保存しました: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ HMM用メタデータ保存エラー: {e}")
            return None

    def save_merged_data(self, merged_df: pd.DataFrame, 
                         teacher_csv_name: str, pred_csv_name: str) -> Optional[Path]:
        """
        マージされたDataFrameをCSVファイルとして保存します。

        Args:
            merged_df (pd.DataFrame): 保存するマージ済みDataFrame。
            teacher_csv_name (str): 元の教師データCSVファイル名（出力ファイル名生成用）。
            pred_csv_name (str): 元の予測CSVファイル名（出力ファイル名生成用）。

        Returns:
            Optional[Path]: 保存されたCSVファイルのパス。エラー時はNone。
        """
        print(f"\n--- マージ結果保存 ---")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # ファイル名から拡張子と不要な部分を除去
            teacher_base = Path(teacher_csv_name).stem.replace("tennis_features_", "").replace("features_", "")
            pred_base = Path(pred_csv_name).stem.replace("lstm_predictions_", "").replace("predictions_", "")

            output_filename = f"merged_{teacher_base}_vs_{pred_base}_{timestamp}.csv"
            output_path = self.output_dir / output_filename
            
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ マージ済みデータを保存しました: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ 結果保存エラー: {e}")
            return None

    def run_merge_pipeline(self,
                           teacher_csv_path: Optional[Path] = None,
                           prediction_csv_path: Optional[Path] = None,
                           teacher_true_label_col: str = 'label', # feature_extractor.py のデフォルト出力列名
                           prediction_pred_label_col: str = 'predicted_phase', # predict_lstm_model.py の想定出力列名
                           merge_keys: Optional[List[str]] = None
                           ):
        """
        マージ処理の完全なパイプラインを実行します。
        ファイルパスが指定されていない場合は、ユーザーに選択を促します。
        """
        print("======= 教師データとLSTM予測のマージ処理開始 =======")

        # 1. 教師データCSVの選択と読み込み
        if teacher_csv_path is None:
            teacher_csv_path = self._select_file_dialog(
                title="教師データCSVファイル選択 (feature_extractor.py 出力)",
                directory=self.base_dir / "features", # feature_extractor.py の出力先想定
                pattern="tennis_features_*.csv"
            )
        teacher_df = self.load_csv(teacher_csv_path, "教師データCSV")
        if teacher_df is None:
            print("❌ 教師データCSVの読み込みに失敗しました。処理を中止します。")
            return

        # 教師データからラベル情報を取得試行
        hmm_labels_info = self._load_teacher_metadata_labels(teacher_csv_path)
        if hmm_labels_info[0] is None: # ラベルリストがNoneなら取得失敗
            print(f"ℹ️ 教師メタデータからラベル情報を取得できなかったため、教師データCSVの '{teacher_true_label_col}' 列から派生させます。")
            hmm_labels_info = self._derive_labels_from_df(teacher_df, teacher_true_label_col)

        if hmm_labels_info[0] is None:
            print(f"❌ HMM用ラベル情報の取得/派生に失敗しました。メタデータ作成をスキップします。")
            # この後も処理は続行するが、メタデータは保存されない

        # 2. LSTM予測CSVの選択と読み込み
        if prediction_csv_path is None:
            prediction_csv_path = self._select_file_dialog(
                title="LSTM予測CSVファイル選択 (predict_lstm_model.py 出力)",
                directory=self.base_dir / "predictions", # predict_lstm_model.py の出力先想定
                pattern="*.csv" # パターンを変更
            )
        prediction_df = self.load_csv(prediction_csv_path, "LSTM予測CSV")
        if prediction_df is None:
            print("❌ LSTM予測CSVの読み込みに失敗しました。処理を中止します。")
            return

        # 3. データのマージ
        merged_df = self.merge_data(teacher_df, prediction_df, 
                                    teacher_true_label_col, prediction_pred_label_col,
                                    merge_keys)
        if merged_df is None or merged_df.empty:
            print("❌ データマージに失敗したか、結果が空です。処理を中止します。")
            return
            
        # 4. マージ結果の保存
        saved_merged_path = self.save_merged_data(merged_df, teacher_csv_path.name, prediction_csv_path.name)

        # 5. HMM用メタデータの保存 (ラベル情報が取得できていれば)
        if saved_merged_path and hmm_labels_info[0] is not None:
            self.save_hmm_metadata(hmm_labels_info, saved_merged_path)
        elif not hmm_labels_info[0]:
            print(f"ℹ️ HMM用ラベル情報がなかったため、HMMメタデータの保存はスキップされました。")


        print("\n🎉 マージ処理完了！")


if __name__ == "__main__":
    # --- 設定項目 ---
    # 教師データCSV (feature_extractor.pyの出力) が保存されているディレクトリ
    # このスクリプトからの相対パス、または絶対パスで指定
    BASE_TRAINING_DATA_DIR = Path("./training_data") 

    # マージ済みファイルの出力先ディレクトリ名 (BASE_TRAINING_DATA_DIR 内)
    # デフォルトは "merged_predictions" ですが、ここで明示的に設定します。
    MERGED_PREDICTIONS_DIR_NAME = "merged_predictions"

    # 教師データCSV内の「真のラベル」が含まれる列名
    # feature_extractor.py のデフォルト出力では 'label'
    TEACHER_TRUE_LABEL_COLUMN = 'label'

    # LSTM予測CSV内の「予測されたラベル」が含まれる列名
    # predict_lstm_model.py の出力に合わせて変更してください
    PREDICTION_PREDICTED_LABEL_COLUMN = 'predicted_phase' 

    # マージに使用するキー列名のリスト
    # 両方のCSVファイルに共通して存在し、行を一意に特定できる列を指定
    # 例: ['video_name', 'frame_number'] や ['video_name', 'original_frame_number']
    # `frame_number` が feature_extractor.py で再割り当てされたインデックスの場合、
    # `original_frame_number` (元動画のフレーム番号) があればそちらが安定する可能性があります。
    # LSTM予測CSVも同じ基準のフレーム番号を持っていることを確認してください。
    MERGE_KEYS = ['video_name', 'frame_number'] 
    # ----------------

    merger = MergeFeaturesAndPredictions(
        output_dir_name=MERGED_PREDICTIONS_DIR_NAME,
        base_dir=BASE_TRAINING_DATA_DIR
    )
    
    try:
        # パイプライン実行
        # 特定のファイルを直接指定する場合は、以下のようにパスを渡します。
        # teacher_file = BASE_TRAINING_DATA_DIR / "features" / "tennis_features_YYYYMMDD_HHMMSS.csv"
        # prediction_file = BASE_TRAINING_DATA_DIR / "predictions" / "lstm_predictions_some_model_YYYYMMDD_HHMMSS.csv"
        # merger.run_merge_pipeline(
        #     teacher_csv_path=teacher_file,
        #     prediction_csv_path=prediction_file,
        #     teacher_true_label_col=TEACHER_TRUE_LABEL_COLUMN,
        #     prediction_pred_label_col=PREDICTION_PREDICTED_LABEL_COLUMN,
        #     merge_keys=MERGE_KEYS
        # )
        
        # ファイル選択をユーザーに促す場合
        merger.run_merge_pipeline(
            teacher_true_label_col=TEACHER_TRUE_LABEL_COLUMN,
            prediction_pred_label_col=PREDICTION_PREDICTED_LABEL_COLUMN,
            merge_keys=MERGE_KEYS
        )

    except KeyboardInterrupt:
        print("\n操作がキャンセルされました。")
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

