import json
import warnings # warnings は既に import されている想定
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report # 追加
import joblib # 追加
from hmmlearn import hmm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# hmmlearn の将来の変更に関する警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning, module='hmmlearn.base')

class HMMSupervisedPostprocessor:
    def __init__(self, 
                 transition_priors=None, forbidden_transitions=None, initial_state_priors=None,
                 verbose=False, random_state=None, base_smoothing_value=1.0):
        """
        HMMPostprocessorを初期化します。
        状態関連の引数 (states, label_to_int, int_to_label) は削除され、
        load_data時にメタデータから設定されます。

        Args:
            transition_priors (Optional[Dict[Tuple[str, str], float]]): 特定の遷移に対する事前確率の重み。
            forbidden_transitions (Optional[List[Tuple[str, str]]]): 禁止する遷移のペアのリスト。
            initial_state_priors (Optional[Dict[str, float]]): 初期状態確率の事前情報（現在は未使用）。
            verbose (bool): 詳細情報を表示するかどうか。
            random_state (Optional[int]): 乱数シード。
            base_smoothing_value (float): 遷移確率と出力確率の計算時に使用する基本的なスムージング値。
        """
        self.df_loaded = None
        self.label_to_int = {}
        self.int_to_label = {}
        self.n_states = 0
        self.states = []
        self.hmm_model = None
        self.trained = False
        self.final_mask_for_hmm = None
        self.valid_observations_int = None
        self.valid_true_labels_int = None

        # 追加された引数をインスタンス変数に設定
        self.verbose = verbose
        self.random_state = random_state
        self.base_smoothing_value = base_smoothing_value
        self.forbidden_transitions_str = forbidden_transitions if forbidden_transitions is not None else []
        self.transition_priors_str = transition_priors if transition_priors is not None else {}
        # initial_state_priors は現在 train メソッドで使用されていないため、必要に応じて設定
        self.initial_state_priors_str = initial_state_priors if initial_state_priors is not None else {}


    def _derive_labels_from_df(self, df, label_source_columns):
        """
        DataFrameからラベル情報を派生させます。
        メタデータJSONがない場合に使用されますが、推奨されません。
        """
        all_labels = set()
        for col in label_source_columns:
            if col in df.columns:
                for label in df[col].unique():
                    # numpy.ndarray型の場合は要素ごとに追加
                    if hasattr(label, "__iter__") and not isinstance(label, str):
                        for l in label:
                            all_labels.add(str(l))
                    else:
                        all_labels.add(str(label))
            else:
                print(f"⚠️ 警告: ラベルソースカラム '{col}' がDataFrameに存在しません。")

        if not all_labels:
            raise ValueError("ラベルソースカラムから有効なラベルを抽出できませんでした。")

        sorted_labels = sorted(list(all_labels))
        label_to_int_map = {label: i for i, label in enumerate(sorted_labels)}
        int_to_label_map = {i: label for i, label in enumerate(sorted_labels)}
        return sorted_labels, label_to_int_map, int_to_label_map

    def load_data(self, data_csv_path: Path, pred_col_name: str, 
                  true_col_name: Optional[str] = None, 
                  metadata_json_path: Optional[Path] = None):
        """
        LSTM予測結果とオプションで真のラベルを含むCSV、およびメタデータJSONを読み込み、HMM処理用のデータを準備します。
        HMMの状態定義は、メタデータJSONから読み込むか、事前にロードされたモデル情報を使用します。

        Args:
            data_csv_path (Path): LSTMによる予測結果が含まれるCSVファイルのパス。
            pred_col_name (str): LSTMによる予測フェーズ（ラベル）が含まれる列名。
            true_col_name (Optional[str]): 真のフェーズ（ラベル）が含まれる列名。指定しない場合は推論モード。
            metadata_json_path (Optional[Path]): LSTMモデルのメタデータJSONファイルのパス。
                                                 モデルがロードされておらず、ラベル情報がない場合に必須。
        """
        print(f"\n--- データ読み込み: {data_csv_path.name} ---")
        
        labels_already_loaded = bool(self.states and self.label_to_int and self.int_to_label)

        if not labels_already_loaded:
            if not metadata_json_path or not metadata_json_path.exists():
                print(f"❌エラー: メタデータJSONファイル (`metadata_json_path`) が指定されていないか、見つかりません。")
                print(f"  HMMモデルが事前にロードされておらず、ラベル情報が未設定の場合、状態定義のためにメタデータが必須です。")
                return False
            
            try:
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                phase_labels_from_meta = metadata.get('phase_labels')
                if not phase_labels_from_meta or not isinstance(phase_labels_from_meta, list):
                    print(f"❌エラー: メタデータに 'phase_labels' (リスト型) が見つからないか、無効です。処理を中断します。")
                    return False
                
                self.states = phase_labels_from_meta
                self.label_to_int = {name: i for i, name in enumerate(self.states)}
                self.int_to_label = {i: name for i, name in enumerate(self.states)}
                self.n_states = len(self.states)

                if self.n_states == 0:
                    print(f"❌エラー: メタデータから有効なラベルが見つかりませんでした ('phase_labels'が空)。処理を中断します。")
                    return False
                print(f"✅ メタデータからラベル情報をロード: {metadata_json_path.name}")
            except Exception as e:
                print(f"❌ メタデータ読み込みエラー: {e}")
                return False
        else:
            print(f"ℹ️ HMMモデルまたは以前の処理でロード済みのラベル情報を使用します。状態数: {self.n_states}")
            if self.verbose:
                print(f"  label_to_int (既存): {self.label_to_int}")

        if self.n_states == 0: # このチェックはラベルロード後に行う
            print(f"❌エラー: HMMの状態数が0です。ラベル情報が正しく読み込まれているか確認してください。")
            return False
        
        if self.verbose and not labels_already_loaded : # メタデータからロードした場合のみ表示（重複を避ける）
            print(f"ℹ️ HMM状態数 (ラベル種類数): {self.n_states}")
            print(f"ℹ️ ラベル -> 整数マッピング (label_to_int): {self.label_to_int}")
            print(f"ℹ️ 整数 -> ラベルマッピング (int_to_label): {self.int_to_label}")
            
        try:
            self.df_loaded = pd.read_csv(data_csv_path)
            if pred_col_name not in self.df_loaded.columns:
                print(f"❌エラー: 指定された予測フェーズ列 '{pred_col_name}' がCSVに存在しません。")
                return False
            
            if true_col_name and true_col_name not in self.df_loaded.columns:
                print(f"❌エラー: 指定された真のフェーズ列 '{true_col_name}' がCSVに存在しません。")
                return False
            
            # --- 予測ラベルの処理 ---
            pred_notna_mask = self.df_loaded[pred_col_name].notna()
            pred_notempty_mask = (self.df_loaded[pred_col_name].astype(str).str.strip() != "")
            pred_isin_meta_mask = self.df_loaded[pred_col_name].isin(self.label_to_int.keys())
            valid_pred_mask = pred_notna_mask & pred_notempty_mask & pred_isin_meta_mask

            final_mask = valid_pred_mask
            self.valid_true_labels_int = None # デフォルトはNone

            if true_col_name:
                # --- 真ラベルの処理 ---
                true_notna_mask = self.df_loaded[true_col_name].notna()
                true_labels_numeric = pd.to_numeric(self.df_loaded[true_col_name], errors='coerce')
                true_isin_range_mask = true_labels_numeric.isin(range(self.n_states))
                valid_true_mask = true_labels_numeric.notna() & true_isin_range_mask
                final_mask = valid_pred_mask & valid_true_mask
                
                if self.verbose:
                    print(f"  真ラベル列 '{true_col_name}':")
                    print(f"    - Original Not NA: {true_notna_mask.sum()} 行")
                    print(f"    - Numeric and Not NA: {true_labels_numeric.notna().sum()} 行")
                    print(f"    - Is in HMM state index range (0 to {self.n_states-1}): {true_isin_range_mask.sum()} 行")
                    print(f"    - Valid True Mask (Numeric, Not NA, AND In Range): {valid_true_mask.sum()} 行")
            
            self.final_mask_for_hmm = final_mask
            
            if self.verbose:
                print(f"\n--- デバッグ情報: マスク詳細 ---")
                print(f"  予測列 '{pred_col_name}':")
                print(f"    - Not NA: {pred_notna_mask.sum()} 行")
                print(f"    - Not Empty (after str conversion): {pred_notempty_mask.sum()} 行")
                print(f"    - Is in Meta Labels (label_to_int.keys): {pred_isin_meta_mask.sum()} 行")
                print(f"    - Valid Pred Mask (上記全てAND): {valid_pred_mask.sum()} 行")
                if true_col_name:
                    print(f"  Final Mask (Valid Pred AND Valid True): {final_mask.sum()} 行")
                else:
                    print(f"  Final Mask (Valid Pred Only): {final_mask.sum()} 行")

            if not final_mask.any():
                errmsg = "有効な観測データが見つかりませんでした。"
                if true_col_name:
                    errmsg = "有効な観測データおよび真のラベルデータが見つかりませんでした。"
                print(f"❌エラー: {errmsg}")
                # (エラー時の詳細デバッグ情報)
                if self.verbose:
                    print(f"\n--- エラー発生時の追加デバッグ情報 ---")
                    print(f"  処理対象CSV: {data_csv_path.name}")
                    print(f"  予測列名: '{pred_col_name}', 真ラベル列名: '{true_col_name}'") # pred_col_name, true_col_name
                    print(f"  メタデータ由来のラベルセット: {set(self.label_to_int.keys())}")
                    
                    if pred_col_name in self.df_loaded: # pred_col_name
                        csv_pred_labels_unique_at_error = self.df_loaded[pred_col_name].dropna().unique() # pred_col_name
                        print(f"  CSV予測列 '{pred_col_name}' のユニークラベル (NaN除外、エラー時): {csv_pred_labels_unique_at_error}") # pred_col_name
                        pred_not_in_meta_at_error = {lbl for lbl in csv_pred_labels_unique_at_error if lbl not in self.label_to_int.keys()}
                        if pred_not_in_meta_at_error:
                             print(f"    ⚠️ 予測ラベルのうちメタデータに存在しないもの (エラー時): {pred_not_in_meta_at_error}")
                    else:
                        print(f"  ⚠️ 予測列 '{pred_col_name}' がDataFrameに存在しません。") # pred_col_name

                    if true_col_name in self.df_loaded: # true_col_name
                        csv_true_labels_unique_at_error = self.df_loaded[true_col_name].dropna().unique() # true_col_name
                        print(f"  CSV真ラベル列 '{true_col_name}' のユニークラベル (NaN除外、エラー時): {csv_true_labels_unique_at_error}") # true_col_name
                        true_not_in_meta_at_error = {lbl for lbl in csv_true_labels_unique_at_error if lbl not in self.label_to_int.keys()}
                        if true_not_in_meta_at_error:
                             print(f"    ⚠️ 真ラベルのうちメタデータに存在しないもの (エラー時): {true_not_in_meta_at_error}")
                    else:
                        print(f"  ⚠️ 真ラベル列 '{true_col_name}' がDataFrameに存在しません。") # true_col_name
                        
                    print(f"  フィルタリング条件:")
                    print(f"    1. 値がNaNでないこと")
                    print(f"    2. 値が空文字列でないこと")
                    print(f"    3. 値がメタデータのラベルセットに含まれていること")
                    print(f"  上記の条件を予測列と真ラベル列の両方で満たす行が0件でした。")
                return False

            # 予測ラベルをHMM状態インデックスに変換
            self.valid_observations_int = self.df_loaded.loc[final_mask, pred_col_name].map(self.label_to_int).values
            
            if true_col_name:
                # 真ラベル(数値)をHMM状態インデックスとして使用 (既に数値変換・範囲チェック済み)
                # final_mask は valid_true_mask を含んでいるので、true_labels_numeric[final_mask] でOK
                self.valid_true_labels_int = true_labels_numeric[final_mask].astype(int).values


            if pd.isna(self.valid_observations_int).any():
                print(f"⚠️ 警告: 予測ラベルのHMM状態インデックスへのマッピングでNaN値が発生しました。")
                missing_pred_labels = self.df_loaded.loc[final_mask, pred_col_name][pd.isna(self.valid_observations_int)].unique()
                print(f"  マッピングできなかった予測ラベル: {missing_pred_labels}")
                print(f"  利用可能なラベル (label_to_int keys): {list(self.label_to_int.keys())}")
                print(f"❌エラー: 予測ラベルのマッピングに失敗しました。処理を中断します。")
                return False
                
            if true_col_name and self.valid_true_labels_int is not None and pd.isna(self.valid_true_labels_int).any():
                print(f"⚠️ 警告: 真ラベルのHMM状態インデックスへの変換で予期せぬNaN値が発生しました。")
                print(f"❌エラー: 真ラベルの変換に失敗しました。処理を中断します。")
                return False

            if len(self.valid_observations_int) == 0:
                 print(f"❌エラー: 観測シーケンス({len(self.valid_observations_int)}点)が空です。処理を中断します。")
                 return False
            
            if true_col_name and self.valid_true_labels_int is not None:
                if len(self.valid_true_labels_int) == 0:
                    print(f"❌エラー: 真ラベルシーケンス({len(self.valid_true_labels_int)}点)が空です。処理を中断します。")
                    return False
                if len(self.valid_observations_int) != len(self.valid_true_labels_int):
                    print(f"❌エラー: 観測シーケンス({len(self.valid_observations_int)}点)と真ラベルシーケンス({len(self.valid_true_labels_int)}点)の長さが一致しません。")
                    return False
            
            if len(self.valid_observations_int) < 2: 
                 print(f"⚠️警告: シーケンス長 ({len(self.valid_observations_int)}) が短すぎます (2未満)。HMMパラメータの計算/平滑化ができない可能性があります。")
                 # 学習時はエラーだが、平滑化時は許容される場合もあるので警告に留める。train側で再度チェック。
                 # return False # ここでは return しない

            print_msg = f"✅ データ準備完了: {len(self.valid_observations_int)}点の有効な観測シーケンス。"
            if true_col_name and self.valid_true_labels_int is not None:
                print_msg = f"✅ データ準備完了: {len(self.valid_observations_int)}点の有効な観測/真ラベルシーケンス。"
            print(print_msg)

            if self.verbose:
                if true_col_name and self.valid_true_labels_int is not None:
                    print(f"  最初の5つの真ラベル (整数インデックス): {self.valid_true_labels_int[:5]}")
                print(f"  最初の5つの観測ラベル (整数インデックス): {self.valid_observations_int[:5]}")
                if self.int_to_label:
                    if true_col_name and self.valid_true_labels_int is not None:
                        print(f"  最初の5つの真ラベル (文字列): {[self.int_to_label.get(i, 'N/A') for i in self.valid_true_labels_int[:5]]}")
                    print(f"  最初の5つの観測ラベル (文字列): {[self.int_to_label.get(i, 'N/A') for i in self.valid_observations_int[:5]]}")
            return True

        except Exception as e:
            print(f"❌ データ読み込み/準備エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train(self, model_save_path: Optional[Path] = None) -> bool: # model_save_path 引数を追加
        """
        教師あり学習によりHMMのパラメータを計算します。
        オプションで学習済みモデルを保存します。

        Args:
            model_save_path (Optional[Path]): 学習済みモデルを保存するパス。指定しない場合は保存しない。

        Returns:
            bool: パラメータ計算が成功した場合はTrue、そうでない場合はFalse。
        """
        # 入力値の検証
        if self.valid_true_labels_int is None or self.valid_observations_int is None:
            print("❌ エラー: 学習に必要なデータ（真ラベルまたは観測データ）が設定されていません。")
            return False

        # 状態数とデータ一貫性チェック
        unique_labels = np.unique(self.valid_true_labels_int)
        if len(unique_labels) > self.n_states:
            print(f"❌ エラー: 真ラベルの一意値数({len(unique_labels)})が定義済み状態数({self.n_states})を超えています。")
            return False

        print(f"\n--- HMMパラメータ計算開始 (教師あり学習、状態数: {self.n_states}) ---")
        
        # データ分布の確認を追加
        if self.verbose:
            print(f"\n--- 学習データのラベル分布確認 ---")
            unique_labels, counts = np.unique(self.valid_true_labels_int, return_counts=True)
            print(f"  学習データに含まれる状態:")
            for label_int, count in zip(unique_labels, counts):
                label_str = self.int_to_label.get(label_int, f"UNKNOWN({label_int})")
                print(f"    {label_str} (インデックス {label_int}): {count} 件")
            
            print(f"\n  全状態の定義:")
            for i in range(self.n_states):
                label_str = self.int_to_label.get(i, f"UNKNOWN({i})")
                print(f"    {label_str} (インデックス {i})")
            
            missing_states = []
            for i in range(self.n_states):
                if i not in unique_labels:
                    missing_states.append(self.int_to_label.get(i, f"UNKNOWN({i})"))
            if missing_states:
                print(f"\n  ⚠️ 学習データに含まれていない状態: {missing_states}")
                print(f"     これらの状態はHMMの出力確率でスムージング値のみが適用されます。")

        try:
            n_states = self.n_states
            n_symbols = self.n_states

            # 1. 初期状態確率 (startprob_) の計算
            # 真のラベルシーケンスにおける各状態の出現頻度を初期状態確率とする (スムージング適用)
            startprob = np.full(n_states, self.base_smoothing_value) # スムージング値で初期化
            if len(self.valid_true_labels_int) > 0: # シーケンスが空でないことを確認
                unique_initial_states, counts_initial_states = np.unique(self.valid_true_labels_int, return_counts=True)
                for state_idx, count in zip(unique_initial_states, counts_initial_states):
                    # startprob の計算では、シーケンスの最初の状態のみを考慮するのが一般的だが、
                    # ここでは全シーケンス中の状態頻度を初期確率として使う実装になっている。
                    # もし最初の状態のみを考慮するなら、 self.valid_true_labels_int[0] のみを使う。
                    # 現在の実装は、各状態が開始状態となりうる相対的な可能性を示している。
                    # ここでは既存のロジックを維持し、スムージング値を加算する形にする。
                    # startprob[state_idx] += count # 元のロジックに近い形
                    pass # 下で別途計算するため、ここでは何もしない

                # シーケンスの最初の状態に基づいて初期状態確率を計算する (より標準的な方法)
                # ただし、データが複数の独立したシーケンスからなる場合は、各シーケンスの開始状態を考慮する必要がある。
                # ここでは単一の長いシーケンスを想定。
                first_state = self.valid_true_labels_int[0]
                startprob[first_state] += 1 # 最初の状態のカウントを増やす (スムージング値に加算)
            
            # 全シーケンス中の状態頻度に基づいて初期状態確率を計算する場合 (元のコードの意図に近い場合)
            # for state_idx in self.true_labels_sequence_int:
            #     startprob[state_idx] += 1
            startprob /= np.sum(startprob)


            # 2. 遷移確率 (transmat_) の計算
            # transmat[i, j] = P(S_{t+1}=j | S_t=i)
            # スムージングとドメイン知識を適用
            transmat = np.full((n_states, n_states), self.base_smoothing_value) # スムージング値で初期化

            # ドメイン知識: transition_priors を適用 (カウントに加算)
            for (from_label, to_label), prior_value in self.transition_priors_str.items():
                if from_label in self.label_to_int and to_label in self.label_to_int:
                    from_idx = self.label_to_int[from_label]
                    to_idx = self.label_to_int[to_label]
                    transmat[from_idx, to_idx] += prior_value
                    if True:
                        print(f"  遷移事前情報適用: {from_label}({from_idx}) -> {to_label}({to_idx}) に {prior_value} を加算")

            # データからの遷移カウント
            for i in range(len(self.valid_true_labels_int) - 1):
                current_state = self.valid_true_labels_int[i]
                next_state = self.valid_true_labels_int[i+1]
                transmat[current_state, next_state] += 1
            
            # ドメイン知識: forbidden_transitions を適用 (確率を0に)
            # 正規化の前に適用することで、他の遷移への確率の再分配を自然に行う
            for from_label, to_label in self.forbidden_transitions_str:
                if from_label in self.label_to_int and to_label in self.label_to_int:
                    from_idx = self.label_to_int[from_label]
                    to_idx = self.label_to_int[to_label]
                    transmat[from_idx, to_idx] = 0 # 禁止遷移の確率を0に
                    if True:
                        print(f"  禁止遷移適用: {from_label}({from_idx}) -> {to_label}({to_idx}) の確率を0に設定")
            
            row_sums = transmat.sum(axis=1, keepdims=True)
            # ゼロ除算を避ける: 行の全ての遷移が禁止され、かつ事前情報もデータもない場合、row_sumが0になりうる
            # そのような状態からはどこへも遷移できないことになる。
            # base_smoothing_value > 0 であれば、通常は0にならないはず。
            # もし row_sum が0になる場合は、その状態からの遷移確率が一様分布になるとするなどの対策が必要。
            # ここでは、スムージングにより0にならないことを期待する。
            # 万が一0になる場合は、その行の確率を均等に割り振る（ただし禁止遷移は0）などの処理も考えられる。
            # 例えば、 if np.any(row_sums == 0): のようなチェックを入れる。
            # 簡単のため、ここではスムージングでカバーされると仮定。
            transmat = np.divide(transmat, row_sums, out=np.zeros_like(transmat), where=row_sums!=0)
            # もしある行の合計が0だった場合、その行はすべて0になる。
            # そのような状態に一度入ると、次の状態への遷移確率が定義されない。
            # これを避けるため、合計が0の行は一様な確率（ただし禁止遷移は0）にするなどのフォールバックを検討する。
            # 例えば、 for i in range(n_states): if row_sums[i] == 0: transmat[i, :] = 1/n_states (要調整)

            # 3. 出力確率 (emissionprob_) の計算
            # emissionprob[i, k] = P(O_t=k | S_t=i)
            # S_t は真のラベル, O_t はLSTM予測ラベル
            # スムージングを適用
            emissionprob = np.full((n_states, n_symbols), self.base_smoothing_value) # スムージング値で初期化
            for true_state, observed_symbol in zip(self.valid_true_labels_int, self.valid_observations_int):
                emissionprob[true_state, observed_symbol] += 1

            row_sums_emission = emissionprob.sum(axis=1, keepdims=True)
            emissionprob = np.divide(emissionprob, row_sums_emission, out=np.zeros_like(emissionprob), where=row_sums_emission!=0)
              # HMMモデルのインスタンス化とパラメータ設定
            self.hmm_model = hmm.CategoricalHMM(
                n_components=n_states,
                random_state=self.random_state
                # init_params="", params="" はfitを呼ばないため不要
            )
            # 計算済みのパラメータをモデルに直接設定
            self.hmm_model.startprob_ = startprob
            self.hmm_model.transmat_ = transmat
            self.hmm_model.emissionprob_ = emissionprob
            
            # hmmlearnの古いバージョンでは _check() が必要だったが、新しいバージョンでは不要な場合がある
            # self.hmm_model._check() # パラメータの整合性をチェック

            if True:
                print(f"  初期状態確率 (startprob_):\n{self.hmm_model.startprob_}")
                print(f"  遷移確率行列 (transmat_):\n{self.hmm_model.transmat_}")
                print(f"  出力確率行列 (emissionprob_):\n{self.hmm_model.emissionprob_}")
            
            print(f"✅ HMMパラメータ計算完了 (教師あり)。")

            if model_save_path:
                self.save_hmm_model(model_save_path)

            return True
        except Exception as e:
            print(f"❌ HMMパラメータ計算エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

    def smooth(self, observed_sequence_to_smooth: Optional[np.ndarray] = None) -> Optional[np.ndarray]: # 引数追加
        """
        学習済みHMMモデルを使用して、観測シーケンス（LSTM予測）を平滑化（最尤状態系列を予測）します。
        引数で観測シーケンスが与えられない場合は、load_dataで読み込まれた観測シーケンスを使用します。

        Args:
            observed_sequence_to_smooth (Optional[np.ndarray]): 平滑化する観測シーケンス（整数エンコード済み）。
                                                              指定しない場合は self.observed_sequence を使用。

        Returns:
            Optional[np.ndarray]: 平滑化された状態シーケンス（整数エンコード済み）。モデルがない場合やエラー時はNone。
        """
        if self.hmm_model is None:
            print("❌エラー: HMMモデルのパラメータが計算されていません。trainを先に実行してください。")
            return None
        if self.valid_observations_int is None:
            print("❌エラー: 平滑化対象の観測シーケンスがありません。")
            return None

        target_sequence = observed_sequence_to_smooth
        if target_sequence is None:
            target_sequence = self.valid_observations_int
        
        if target_sequence is None:
            print("❌ エラー: 平滑化対象の観測シーケンスがありません。")
            return None

        if True:
            print(f"\n--- HMMによる予測平滑化開始 ---")
        
        try:
            # 観測シーケンスは (n_samples, 1) の形状である必要がある
            if target_sequence.ndim == 1:
                reshaped_sequence = target_sequence.reshape(-1, 1)
            else:
                reshaped_sequence = target_sequence
            
            logprob, smoothed_states_int = self.hmm_model.decode(reshaped_sequence, algorithm="viterbi")
            
            if True:
                print(f"✅ 予測平滑化完了: {len(smoothed_states_int)}点。対数確率: {logprob:.2f}")
            return smoothed_states_int
        except Exception as e:
            print(f"❌ HMM予測エラー: {e}")
            return None

    def add_smoothed_results_to_df(self, smoothed_sequence_labels):
        """平滑化された予測結果を元のDataFrameに追加します。"""
        if self.final_mask_for_hmm is None:
            print("❌エラー: HMM処理用のマスク (self.final_mask_for_hmm) が設定されていません。")
            return False

        num_masked_rows = self.final_mask_for_hmm.sum()

        if len(smoothed_sequence_labels) != num_masked_rows:
            print(f"❌エラー: 平滑化シーケンス長 ({len(smoothed_sequence_labels)}) とマスクされた行数 ({num_masked_rows}) が一致しません。")
            print("⚠️ 平滑化結果のDataFrameへの追加に失敗しました。結果は保存されません。")
            return False

        # 新しい列をNaNまたは適切なプレースホルダーで初期化
        self.df_loaded['hmm_predicted_phase'] = pd.NA # pandas 1.0+ の場合。古い場合は np.nan

        # マスクを使用して正しい位置に平滑化されたラベルを割り当てる
        # smoothed_sequence_labels は numpy 配列またはリストを想定
        self.df_loaded.loc[self.final_mask_for_hmm, 'hmm_predicted_phase'] = smoothed_sequence_labels
        
        # (オプション) hmm_predicted_phase_int も保存する場合
        # smoothed_sequence は整数のシーケンスなので、それも保存できる
        # self.df_loaded['hmm_predicted_phase_int'] = pd.NA
        # self.df_loaded.loc[self.final_mask_for_hmm, 'hmm_predicted_phase_int'] = smoothed_sequence # smoothed_sequence は平滑化された状態の整数列

        print("✅ 平滑化結果をDataFrameに 'hmm_predicted_phase' 列として追加しました。")
        return True

    def save_results(self, output_df: pd.DataFrame, input_csv_path: Path, output_base_dir: Optional[Path] = None) -> Optional[Path]:
        """
        処理結果のDataFrameをCSVファイルとして保存します。

        Args:
            output_df (pd.DataFrame): 保存するDataFrame。
            input_csv_path (Path): 元の入力CSVファイルのパス（出力ファイル名生成用）。
            output_base_dir (Optional[Path]): 出力ファイルの保存先ベースディレクトリ。指定しない場合はデフォルトの場所。

        Returns:
            Optional[Path]: 保存されたCSVファイルのパス。エラー時はNone。
        """
        print(f"\n--- 結果保存 ---")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = input_csv_path.stem
            # "predictions" を "hmm_processed" に置換、または末尾に追加
            if "predictions" in base_name:
                 output_filename_stem = base_name.replace("predictions", "hmm_processed")
            else:
                 output_filename_stem = f"{base_name}_hmm_processed"
            
            output_filename = f"{output_filename_stem}_{timestamp}.csv"

            if output_base_dir:
                target_dir = output_base_dir
            else:
                target_dir = Path("./training_data/hmm_processed_predictions") # デフォルト
            
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / output_filename
            
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ HMM後処理済み予測結果を保存しました: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ 結果保存エラー: {e}")
            return None

    def _select_file_dialog(self, title: str, directory: Path, pattern: str) -> Optional[Path]:
        """簡易的なファイル選択ダイアログ（コンソール）"""
        print(f"\n=== {title} ===")
        files = sorted(list(directory.glob(pattern)))
        if not files:
            print(f"❌ {pattern} に一致するファイルが見つかりません in {directory}")
            return None

        for i, f_path in enumerate(files, 1):
            print(f"  {i}. {f_path.name} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(files)}), 0でスキップ: ").strip()
            choice_num = int(choice)
            if choice_num == 0:
                return None
            if 1 <= choice_num <= len(files):
                return files[choice_num - 1]
            else:
                print("無効な選択です。")
                return None
        except ValueError:
            print("無効な入力です。")
            return None

    def save_hmm_model(self, model_output_path: Path) -> bool:
        """学習済みのHMMモデルと関連情報（ラベルマッピング）をファイルに保存します。"""
        if self.hmm_model is None or not self.label_to_int or not self.int_to_label:
            print("❌ エラー: 保存する学習済みHMMモデルまたはラベル情報が存在しません。")
            return False
        try:
            model_data = {
                'model_params': {
                    'startprob_': self.hmm_model.startprob_,
                    'transmat_': self.hmm_model.transmat_,
                    'emissionprob_': self.hmm_model.emissionprob_,
                    'n_components': self.hmm_model.n_components,
                    'random_state': self.hmm_model.random_state
                },
                'label_to_int': self.label_to_int,
                'int_to_label': self.int_to_label,
                'labels': self.states,
                'n_components': self.n_states, # 保存時のn_componentsも記録
                'base_smoothing_value': self.base_smoothing_value, # 参考情報として
                'forbidden_transitions_str': self.forbidden_transitions_str, # 参考情報
                'transition_priors_str': self.transition_priors_str # 参考情報
            }
            model_output_path.parent.mkdir(parents=True, exist_ok=True) # joblibがやってくれるはず -> 明示的に作成
            joblib.dump(model_data, model_output_path)
            if True:
                print(f"✅ HMMモデルと関連情報を保存しました: {model_output_path}")
            return True
        except Exception as e:
            print(f"❌ HMMモデルの保存中にエラーが発生しました: {e}")
            return False

    def load_hmm_model(self, model_input_path: Path) -> bool:
        """ファイルから学習済みHMMモデルと関連情報を読み込みます。"""
        try:
            if not model_input_path.exists():
                print(f"❌ エラー: HMMモデルファイルが見つかりません: {model_input_path}")
                return False
                
            model_data = joblib.load(model_input_path)
            
            self.label_to_int = model_data['label_to_int']
            self.int_to_label = model_data['int_to_label']
            self.states = model_data['labels']
            self.n_states = model_data['n_components']
              # HMMモデルの再構築
            params = model_data['model_params']
            self.hmm_model = hmm.CategoricalHMM(
                n_components=params['n_components'],
                random_state=params.get('random_state') # 古い保存形式互換
            )
            self.hmm_model.startprob_ = params['startprob_']
            self.hmm_model.transmat_ = params['transmat_']
            self.hmm_model.emissionprob_ = params['emissionprob_']
            
            # 参考情報を復元（オプション）
            self.base_smoothing_value = model_data.get('base_smoothing_value', self.base_smoothing_value)
            self.forbidden_transitions_str = model_data.get('forbidden_transitions_str', self.forbidden_transitions_str)
            self.transition_priors_str = model_data.get('transition_priors_str', self.transition_priors_str)

            if True:
                print(f"✅ HMMモデルと関連情報を読み込みました: {model_input_path}")
                print(f"  読み込み後のラベルマッピング: {self.label_to_int}")
                print(f"  読み込み後の状態数: {self.n_states}")
            return True
        except Exception as e:
            print(f"❌ HMMモデルの読み込み中にエラーが発生しました: {e}")
            return False

    def evaluate_predictions(self, 
                             true_labels_int: np.ndarray, 
                             predicted_labels_int: np.ndarray, 
                             target_name: str):
        """予測結果の評価指標（Accuracy, Precision, Recall, F1-score）を計算し表示します。
        引数:
            true_labels_int (np.ndarray): 真のラベルの整数エンコードシーケンス。
            predicted_labels_int (np.ndarray): 予測ラベルの整数エンコードシーケンス。
            target_name (str): 評価対象の名前（例: "LSTM予測" または "HMM平滑化後予測"）。
        """
        if true_labels_int is None or predicted_labels_int is None: # self.int_to_label のチェックは後段へ移動
            print(f"⚠️ {target_name} の評価に必要なラベルデータが不足しています。スキップします。")
            return

        if self.verbose:
            print(f"\n--- {target_name} 評価デバッグ情報 ---")
            print(f"  真ラベル配列長: {len(true_labels_int)}")
            print(f"  予測ラベル配列長: {len(predicted_labels_int)}")
            print(f"  真ラベル unique値: {np.unique(true_labels_int)}")
            print(f"  予測ラベル unique値: {np.unique(predicted_labels_int)}")
            if self.int_to_label:
                try:
                    true_labels_str_unique = [self.int_to_label.get(i, f"UNMAPPED_INT({i})") for i in np.unique(true_labels_int)]
                    pred_labels_str_unique = [self.int_to_label.get(i, f"UNMAPPED_INT({i})") for i in np.unique(predicted_labels_int)]
                    print(f"  真ラベル unique値 (文字列マッピング試行後): {true_labels_str_unique}")
                    print(f"  予測ラベル unique値 (文字列マッピング試行後): {pred_labels_str_unique}")
                except Exception as e: # pylint: disable=broad-except
                    print(f"  警告: int_to_label を使用したデバッグ情報表示中にエラー: {e}")
            else:
                print("  int_to_label が未設定です。")

        min_len = min(len(true_labels_int), len(predicted_labels_int))
        if min_len == 0: # min_len のチェックを修正
            print(f"⚠️ {target_name} の評価対象データが0件です。スキップします。")
            return
            
        true_labels_int_aligned = true_labels_int[:min_len]
        predicted_labels_int_aligned = predicted_labels_int[:min_len]
        
        if self.verbose:
            print(f"  評価対象データ長 (整合後): {min_len}")

        # load_dataでラベルは共通の整数空間にマップされるため、
        # LSTM予測と真ラベルの数値空間が異なるという特別処理は不要。
        # if target_name == "LSTM予測 (元の予測)":
        #     ... (このブロック全体を削除) ...

        # NaNや無効値のチェック - データをfloat型に変換してNaNチェック実行
        # 整数配列のはずなので、NaNチェックは通常不要だが、念のため残す
        true_labels_int_aligned_float = true_labels_int_aligned.astype(float)
        predicted_labels_int_aligned_float = predicted_labels_int_aligned.astype(float)

        if np.isnan(true_labels_int_aligned_float).any() or np.isnan(predicted_labels_int_aligned_float).any():
            print(f"⚠️ {target_name} の評価対象データにNaN値が含まれています。スキップします。")
            if self.verbose:
                print(f"    NaN in true_labels: {np.isnan(true_labels_int_aligned_float).sum()}")
                print(f"    NaN in predicted_labels: {np.isnan(predicted_labels_int_aligned_float).sum()}")
            return
        
        if not self.int_to_label: # classification_report でラベル名が必要なため、ここでチェック
            print(f"⚠️ {target_name} の評価に必要なラベル名マッピング (self.int_to_label) が不足しています。レポートの一部が制限される可能性があります。")
            # 評価自体は続行可能

        try:
            accuracy = accuracy_score(true_labels_int_aligned, predicted_labels_int_aligned)
            
            present_labels_int = sorted(list(set(true_labels_int_aligned) | set(predicted_labels_int_aligned)))
            
            report_kwargs = {'labels': present_labels_int, 'zero_division': 0}
            if self.int_to_label:
                # マッピングが存在しない整数ラベルがある場合、そのラベルは整数として表示される
                target_names_for_report = [self.int_to_label.get(i, str(i)) for i in present_labels_int]
                report_kwargs['target_names'] = target_names_for_report
            else:
                # int_to_label がない場合は、ラベルは整数として表示される (デフォルトの動作)
                pass


            report = classification_report(true_labels_int_aligned, predicted_labels_int_aligned, **report_kwargs)
            
            print(f"\n--- {target_name} の評価結果 ---")
            print(f"  対象データ数: {len(true_labels_int_aligned)}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Classification Report:\n{report}")
        except Exception as e:
            print(f"❌ {target_name} の評価中にエラーが発生しました: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def run_postprocessing_pipeline(self,
                                    lstm_pred_csv_path: Optional[Path] = None,
                                    true_phase_column_name: Optional[str] = 'true_phase', # Optionalに変更したが、パイプラインでは通常指定
                                    lstm_metadata_json_path: Optional[Path] = None, # このパスが重要になる
                                    predicted_phase_column_name: str = 'predicted_phase',
                                    hmm_model_save_path: Optional[Path] = None, 
                                    hmm_model_load_path: Optional[Path] = None
                                    ):        
        print("=== HMM 後処理パイプライン開始 (教師あり学習モード) ===")
        
        original_verbose_state = self.verbose 
        self.verbose = True 

        model_loaded_successfully = False
        if hmm_model_load_path:
            if True:
                print(f"\n--- HMMモデル読み込み試行: {hmm_model_load_path} ---")
            if self.load_hmm_model(hmm_model_load_path):
                model_loaded_successfully = True
            else:
                print(f"⚠️ HMMモデルの読み込みに失敗しました。ファイルパス: {hmm_model_load_path}")
                print("  通常の学習パイプラインを続行します。")

        selected_csv_path = lstm_pred_csv_path
        if not selected_csv_path:
            selected_csv_path = self._select_file_dialog(
                title="学習データCSVファイル選択 (LSTM予測と真ラベルを含むマージ済みファイル)",
                directory=Path("./lgbm_models"),
                pattern="*.csv"
            )
        if not selected_csv_path:
            print("❌ CSVファイルが選択されませんでした。処理を中止します。")
            self.verbose = original_verbose_state
            return
        
        selected_metadata_path = lstm_metadata_json_path
        # モデルがロードされず、かつ外部からメタデータパスが提供されなかった場合のみ、ファイル選択を試みる
        if not model_loaded_successfully and not selected_metadata_path:
            print("\n--- LSTMメタデータJSONファイル選択 (必須) ---")
            print("HMMの状態定義とラベルマッピングのためにメタデータJSONファイルが必要です。")
            
            potential_metadata_files = []
            lstm_models_base_dir = Path("./training_data/lstm_models")
            if lstm_models_base_dir.exists():
                found_in_lstm_models = sorted(list(lstm_models_base_dir.rglob("*metadata*.json")))
                potential_metadata_files.extend(found_in_lstm_models)
                if True and found_in_lstm_models:
                    print(f"  {lstm_models_base_dir} およびサブフォルダから {len(found_in_lstm_models)} 個のメタデータ候補を検出。")

            training_data_dir = Path("./training_data")
            if training_data_dir.exists():
                found_in_training_data = sorted([
                    p for p in training_data_dir.glob("*metadata*.json") 
                    if p.is_file() and (not lstm_models_base_dir.exists() or lstm_models_base_dir.resolve() not in p.parents)
                ])
                potential_metadata_files.extend(found_in_training_data)
                if True and found_in_training_data:
                    print(f"  {training_data_dir} 直下から {len(found_in_training_data)} 個のメタデータ候補を検出。")
            
            if potential_metadata_files:
                potential_metadata_files = sorted(list(set(potential_metadata_files)), key=lambda p: str(p))
                print("\n--- LSTMメタデータJSONファイル選択 ---")
                try:
                    current_dir_resolved = Path('.').resolve()
                except FileNotFoundError: 
                    current_dir_resolved = Path('.')
                for i, f_path in enumerate(potential_metadata_files, 1):
                    try:
                        relative_path_str = str(f_path.relative_to(current_dir_resolved))
                    except ValueError: 
                        relative_path_str = str(f_path)
                    print(f"  {i}. {relative_path_str} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
                try:
                    choice = input(f"選択してください (1-{len(potential_metadata_files)}), 0でスキップ: ").strip()
                    choice_num = int(choice)
                    if choice_num == 0:
                        print("⚠️ メタデータファイルが選択されませんでした（スキップされました）。")
                        selected_metadata_path = None
                    elif 1 <= choice_num <= len(potential_metadata_files):
                        selected_metadata_path = potential_metadata_files[choice_num - 1]
                        try:
                            display_path = selected_metadata_path.relative_to(current_dir_resolved)
                        except ValueError:
                            display_path = selected_metadata_path
                        print(f"✅ 選択されたメタデータファイル: {display_path}")
                    else:
                        print("無効な選択です。メタデータファイルは使用されません。")
                        selected_metadata_path = None
                except ValueError:
                    print("無効な入力です。メタデータファイルは使用されません。")
                    selected_metadata_path = None
            else:
                print(f"⚠️ メタデータファイル (*metadata*.json) が見つかりませんでした。")
                selected_metadata_path = None

        # load_data に渡すメタデータパスを決定
        # モデルロード成功時は、モデル内のラベル情報が優先されるが、load_dataはパスを要求する。
        # モデルロード失敗時は、選択または指定されたメタデータパスが必須。
        metadata_path_for_load_data = selected_metadata_path
        if not model_loaded_successfully and not metadata_path_for_load_data:
            print("❌エラー: HMMモデルをロードせず、かつ有効なメタデータファイルも指定/選択されなかったため、処理を中止します。")
            self.verbose = original_verbose_state
            return
        
        # モデルロード成功時でも、load_dataはmetadata_json_pathを要求するため、
        # selected_metadata_pathがNoneであればエラーとするか、ダミーパスを許容する設計が必要。
        # 現状のload_dataはパスの存在をチェックするため、有効なパスが必要。
        # 外部から与えられたlstm_metadata_json_pathをそのまま使う。
        # もしそれがNoneでモデルもロードしない場合は上でエラーになっている。
        # load_data側の修正で、モデルロード済みならmetadata_json_pathがNoneでも許容されるようになった。
        if not metadata_path_for_load_data and model_loaded_successfully:
             print(f"ℹ️ モデルはロードされましたが、外部メタデータパスが指定されていません。ロードされたモデルのラベル情報を使用します。")
             # metadata_path_for_load_data は None のままでOK

        if not self.load_data(selected_csv_path,
                              pred_col_name=predicted_phase_column_name,
                              true_col_name=true_phase_column_name, # パイプラインでは真ラベルありを想定
                              metadata_json_path=metadata_path_for_load_data): # モデルロード済みならNoneでも可
            print("❌ データ読み込みに失敗しました。処理を中断します。")
            self.verbose = original_verbose_state
            return

        if model_loaded_successfully:
            if not self.label_to_int or not self.int_to_label: # load_hmm_modelで設定されるはず
                print("❌エラー: HMMモデルはロードされましたが、ラベル情報が復元されていません。")
                self.verbose = original_verbose_state
                return
            if self.verbose:
                print("ℹ️ ロードされたHMMモデルのラベル情報を使用します。")
                print(f"  label_to_int (from loaded model): {self.label_to_int}")

        if self.valid_observations_int is None or self.valid_true_labels_int is None:
            print("❌ 処理に必要なシーケンスデータ（観測または真ラベル）がありません。")
            self.verbose = original_verbose_state
            return

        # 3. HMMモデルの学習 (モデルがロードされなかった場合)
        if not model_loaded_successfully:
            if self.valid_true_labels_int is None:
                print("❌ HMMモデルの学習に必要な真ラベルシーケンスがありません（モデルもロードされていません）。処理を中止します。")
                return

            if True:
                print(f"\n--- HMMモデル学習 ---")
            if not self.train(model_save_path=hmm_model_save_path): # 保存パスを渡す
                print("❌ HMMモデルの学習に失敗しました。処理を中断します。")
                return
        else:
            if True:
                print(f"ℹ️ 学習済みHMMモデル ({hmm_model_load_path.name}) を使用するため、学習はスキップされました。")

        # 4. LSTM単体での評価 (真ラベルがある場合) - 改善版
        if self.valid_true_labels_int is not None and self.valid_observations_int is not None:
            if self.verbose:
                print(f"\n--- LSTM評価前のデータ確認 ---")
                print(f"  真ラベルシーケンス長: {len(self.valid_true_labels_int)}")
                print(f"  観測シーケンス長: {len(self.valid_observations_int)}")
                print(f"  ラベルマッピング辞書サイズ: {len(self.int_to_label)}")
            
            try:
                self.evaluate_predictions(self.valid_true_labels_int, 
                                          self.valid_observations_int, 
                                          "LSTM予測 (元の予測)")
            except Exception as e:
                print(f"❌ LSTM評価中にエラーが発生しました: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            if self.valid_true_labels_int is None:
                print("⚠️ 真ラベルシーケンスがないため、LSTM評価をスキップします。")
            if self.valid_observations_int is None:
                print("⚠️ 観測シーケンスがないため、LSTM評価をスキップします。")
        
        # 5. HMMによる平滑化
        if True:
            print(f"\n--- HMMによる平滑化 ---")
        smoothed_sequence_int = self.smooth()
        if smoothed_sequence_int is None:
            print("❌ HMMによる平滑化に失敗しました。")

        # 6. HMM適用後の評価 (真ラベルと平滑化結果がある場合)
        if self.valid_true_labels_int is not None and smoothed_sequence_int is not None:
            try:
                self.evaluate_predictions(self.valid_true_labels_int, 
                                          smoothed_sequence_int, 
                                          "HMM平滑化後予測")
            except Exception as e:
                print(f"❌ HMM平滑化後評価中にエラーが発生しました: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        # 7. 結果のDataFrameへの追加と保存 (平滑化結果がある場合)
        if smoothed_sequence_int is not None and self.df_loaded is not None:
            if True:
                print(f"\n--- 平滑化結果をDataFrameに追加 ---")
            
            # 平滑化された整数シーケンスをラベルシーケンスに変換
            smoothed_sequence_labels = np.array([self.int_to_label.get(s, "UNKNOWN_STATE") for s in smoothed_sequence_int])

            if self.add_smoothed_results_to_df(smoothed_sequence_labels):
                if True:
                    print(f"\n--- 結果保存 ---")
                # run_postprocessing_pipeline はデフォルトの出力先を使うため output_base_dir は指定しない
                self.save_results(self.df_loaded, selected_csv_path) 
            else:
                print("⚠️ 平滑化結果のDataFrameへの追加に失敗しました。結果は保存されません。")
        elif smoothed_sequence_int is None:
             print("⚠️ 平滑化が行われなかったため、結果の追加・保存はスキップされました。")

        # パイプラインの最後に元の verbose 状態に戻す
        self.verbose = original_verbose_state
        print("\n=== HMM 後処理パイプライン終了 ===")


if __name__ == "__main__":
    print("=== テニス動画局面分類 LSTM予測結果 HMM後処理ツール (教師あり学習モード) ===")
    
    # --- 設定項目 ---
    TRUE_LABEL_COLUMN = 'label' # CSVファイル内の真ラベルが含まれる列名
    PREDICTED_LABEL_COLUMN = 'predicted_phase' # CSVファイル内のLSTM予測ラベルが含まれる列名
    
    # --- LSTM_METADATA_PATH の動的選択 ---
    base_lstm_models_path = Path("./training_data/lstm_models/")
    available_models = [d for d in base_lstm_models_path.iterdir() if d.is_dir()]

    if not available_models:
        print(f"❌ エラー: {base_lstm_models_path} に利用可能なモデルディレクトリが見つかりません。")
        exit()

    print("\n利用可能なモデルディレクトリ:")
    for i, model_dir in enumerate(available_models):
        print(f"  {i + 1}: {model_dir.name}")

    while True:
        try:
            choice = int(input("使用するモデルディレクトリの番号を選択してください: ")) - 1
            if 0 <= choice < len(available_models):
                selected_model_dir = available_models[choice]
                
                # 選択されたディレクトリ内の .json ファイルを検索
                json_files = list(selected_model_dir.glob('*.json'))

                if not json_files:
                    print(f"❌ エラー: {selected_model_dir} 内に .json ファイルが見つかりません。")
                    # LSTM_METADATA_PATH を None にしておくか、エラー処理を強化
                    LSTM_METADATA_PATH = None 
                elif len(json_files) == 1:
                    LSTM_METADATA_PATH = json_files[0]
                    print(f"メタデータパスとして {LSTM_METADATA_PATH} を使用します。")
                else:
                    print("複数の .json ファイルが見つかりました。使用するファイルを選択してください:")
                    for i, f_path in enumerate(json_files):
                        print(f"  {i + 1}: {f_path.name}")
                    while True:
                        try:
                            file_choice = int(input("メタデータファイルの番号を選択してください: ")) - 1
                            if 0 <= file_choice < len(json_files):
                                LSTM_METADATA_PATH = json_files[file_choice]
                                print(f"メタデータパスとして {LSTM_METADATA_PATH} を使用します。")
                                break
                            else:
                                print("無効な選択です。もう一度入力してください。")
                        except ValueError:
                            print("数値を入力してください。")
                
                # LSTM_METADATA_PATH が設定されたか、エラーで None のままかを確認
                if LSTM_METADATA_PATH and not LSTM_METADATA_PATH.exists(): # このチェックは実際には不要かも（globで見つかった時点で存在するはず）
                     print(f"❌ エラー: {LSTM_METADATA_PATH} が見つかりません。") # 万が一のためのメッセージ
                break
            else:
                print("無効な選択です。もう一度入力してください。")
        except ValueError:
            print("数値を入力してください。")
    # ------------------------------------

    # LSTM_METADATA_PATH = Path("./training_data/lstm_models/your_model_metadata.json") # 例: 実際のパスに置き換えてください
    # LSTM_METADATA_PATH = None # モデルロード時など、外部から指定しない場合はNoneも可

    # 実際の実行前に、LSTM_METADATA_PATH が適切に設定されているか確認するロジックを追加
    # (ただし、HMM_MODEL_LOAD_PATH が指定されている場合は、このパスは必須ではないかもしれない)
    # ここでは、パイプライン内でパスの検証を行うため、事前のexitは削除。

    # HMMモデルの保存パスと読み込みパス
    hmm_models_dir = Path("./training_data/hmm_models")
    hmm_models_dir.mkdir(parents=True, exist_ok=True) # 保存先ディレクトリを作成

    HMM_MODEL_SAVE_PATH = hmm_models_dir / "hmm_model_supervised.joblib"
    # HMM_MODEL_LOAD_PATH = hmm_models_dir / "hmm_model_supervised.joblib" # 必要に応じてコメント解除して使用
    HMM_MODEL_LOAD_PATH = None


    forbidden_transitions_config = [
        ('point_interval', 'rally'),
        ('rally', 'serve_front_ad'),
        ('rally', 'serve_back_ad'),
        ('rally', 'serve_front_deuce'),
        ('rally', 'serve_back_deuce'),
        ('serve_front_deuce', 'serve_back_deuce'),
        ('serve_front_deuce', 'serve_front_ad'),
        ('serve_front_deuce', 'serve_back_ad'),
        ('serve_front_deuce', 'changeover'),    # 不足していた要素を追加
        ('serve_front_ad', 'serve_back_ad'),
        ('serve_front_ad', 'serve_front_deuce'),
        ('serve_front_ad', 'serve_back_deuce'),
        ('serve_front_ad', 'changeover'), 
        ('serve_back_deuce', 'serve_front_deuce'),
        ('serve_back_deuce', 'serve_front_ad'),
        ('serve_back_deuce', 'serve_back_ad'),
        ('serve_back_deuce', 'changeover'),
        ('serve_back_ad', 'serve_front_deuce'),
        ('serve_back_ad', 'serve_front_ad'),
        ('serve_back_ad', 'serve_back_deuce'),
        ('serve_back_ad', 'changeover'),
        ('changeover', 'rally'),
        ('changeover', 'serve_front_ad'),
        ('changeover', 'serve_back_ad'), 
        ('changeover', 'serve_front_deuce'),
        ('changeover', 'serve_back_deuce'),
    ]

    # ------------------------------------

    # ここからパイプライン実行
    postprocessor = HMMSupervisedPostprocessor(
        transition_priors=None, # ここではNoneを指定
        forbidden_transitions=forbidden_transitions_config, # 設定した禁止遷移
        initial_state_priors=None, # 初期状態確率はtrainメソッドで計算
        verbose=True, # 詳細情報を表示
        random_state=42, # 乱数シード
        base_smoothing_value=1.0 # 基本スムージング値
    )

    # run_postprocessing_pipeline メソッドを呼び出す
    postprocessor.run_postprocessing_pipeline(
        lstm_pred_csv_path=None, # ここではファイル選択ダイアログを表示させるためNone
        true_phase_column_name=TRUE_LABEL_COLUMN,
        lstm_metadata_json_path=LSTM_METADATA_PATH, # 動的に選択されたメタデータパス
        predicted_phase_column_name=PREDICTED_LABEL_COLUMN,
        hmm_model_save_path=HMM_MODEL_SAVE_PATH,
        hmm_model_load_path=HMM_MODEL_LOAD_PATH
    )

