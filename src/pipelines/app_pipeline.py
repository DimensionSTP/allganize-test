from typing import Dict, List, Optional, Union
import os

import streamlit as st
import pandas as pd
import requests

from omegaconf import DictConfig

from ..managers import RecommendationManager, ReportManager
from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    @st.cache_data
    def get_categories() -> List[Optional[str]]:
        try:
            items_path = os.path.join(
                config.data_path,
                config.items_name,
            )
            df = pd.read_csv(items_path)
            categories = (
                df[config.category_column_name].dropna().astype(str).unique().tolist()
            )
            return sorted(categories)
        except Exception:
            return []

    def _headers() -> Dict[str, str]:
        header = {"Content-Type": "application/json"}
        if config.api_key:
            header["Authorization"] = f"Bearer {config.api_key}"
        return header

    def remote_recommend(
        input_value: str,
        input_type: str,
        category_value: Optional[str],
    ) -> Optional[str]:
        if not config.remote_api_base_recommend:
            raise RuntimeError("remote_api_base_recommend is not set.")
        payload = {
            "input_value": input_value,
            "input_type": input_type,
            "category_value": category_value,
        }
        r = requests.post(
            f"{config.remote_api_base_recommend}/recommend",
            json=payload,
            headers=_headers(),
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("result", data)

    def remote_report(recommendations: str) -> str:
        if not config.remote_api_base_report:
            raise RuntimeError("remote_api_base_report is not set.")
        payload = {"recommendations": recommendations}
        r = requests.post(
            f"{config.remote_api_base_report}/report",
            json=payload,
            headers=_headers(),
            timeout=240,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("text", "")

    @st.cache_resource(show_spinner=True)
    def get_cached_manager(
        manager_type: str,
    ) -> Optional[Union[RecommendationManager, ReportManager]]:
        if config.remote_mode:
            return None
        setup = SetUp(config)
        manager = setup.get_manager(manager_type=manager_type)
        return manager

    recommendation_manager = get_cached_manager(manager_type="recommendation")
    report_manager = get_cached_manager(manager_type="report")

    st.title("Recipe AI Demo")

    lab_number_recommendation_mode = "Lab number based recommendation"
    ingredients_recommendation_mode = "Ingredients based recommendation"

    option = st.selectbox(
        "Please select the recommendation mode",
        [
            lab_number_recommendation_mode,
            ingredients_recommendation_mode,
        ],
    )

    if option == lab_number_recommendation_mode:
        lab_id = st.text_input("Enter the lab number to recommend for:")
        if recommendation_manager is not None:
            try:
                categories = (
                    recommendation_manager.index.df[
                        recommendation_manager.category_column_name
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                categories = sorted(categories)
            except Exception:
                categories = []
        else:
            categories = get_categories()

        category_options = ["ALL"] + categories
        category_value = st.selectbox(
            "Select a category (use 'ALL' to include all categories):",
            options=category_options,
            index=0,
        )

        recommend_column, reset_column = st.columns([1, 1])
        with recommend_column:
            run = st.button("recommend", type="primary")
        with reset_column:
            reset = st.button("reset")

        if reset:
            st.session_state.clear()
            st.rerun()

        if run:
            lab_id = lab_id.strip()
            category_value = category_value.strip()

            if not lab_id:
                st.warning("Enter the lab number to recommend for.")
            else:
                if isinstance(category_value, str) and category_value.lower() == "all":
                    category_value = None

                with st.spinner("Recommendation in progress..."):
                    try:
                        if recommendation_manager is not None:
                            recommendations = recommendation_manager.recommend(
                                input_value=lab_id,
                                input_type=config.input_mode.lab_id,
                                category_value=category_value,
                            )
                        else:
                            recommendations = remote_recommend(
                                input_value=lab_id,
                                input_type=config.input_mode.lab_id,
                                category_value=category_value,
                            )
                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")
                    else:
                        st.session_state["last_recommendations"] = recommendations
                        st.session_state["last_report"] = None
        if st.session_state.get("last_recommendations") is not None:
            st.subheader("Summary of AI recommendations")
            _rec = st.session_state["last_recommendations"]
            if isinstance(_rec, (dict, list)):
                st.json(_rec)
            elif isinstance(_rec, str) and (
                "<br/>" in _rec or "<strong>" in _rec or "<p>" in _rec
            ):
                st.markdown(
                    _rec,
                    unsafe_allow_html=True,
                )
            else:
                st.write(_rec)
            if st.button("generate report", key="gen_report_lab"):
                with st.spinner("Report generation in progress..."):
                    try:
                        if report_manager is not None:
                            report = report_manager.generate(
                                recommendations=st.session_state[
                                    "last_recommendations"
                                ],
                            )
                        else:
                            report = remote_report(
                                recommendations=st.session_state[
                                    "last_recommendations"
                                ],
                            )
                    except Exception as e:
                        st.error(f"Error during report generation: {e}")
                    else:
                        st.session_state["last_report"] = report
        if st.session_state.get("last_report") is not None:
            st.subheader("Summary of AI report")
            st.write(st.session_state["last_report"])
    elif option == ingredients_recommendation_mode:
        st.write("Enter ingredients one per line:")
        ingredients_input = st.text_area(
            "Ingredients list",
            height=180,
            placeholder="A\nB\nC\n...",
        )

        if recommendation_manager is not None:
            try:
                categories = (
                    recommendation_manager.index.df[
                        recommendation_manager.category_column_name
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                categories = sorted(categories)
            except Exception:
                categories = []
        else:
            categories = get_categories()

        category_options = ["ALL"] + categories
        category_value = st.selectbox(
            "Select a category (use 'ALL' to include all categories):",
            options=category_options,
            index=0,
        )

        recommend_column, reset_column = st.columns([1, 1])
        with recommend_column:
            run = st.button("recommend", type="primary")
        with reset_column:
            reset = st.button("reset")

        if reset:
            st.session_state.clear()
            st.rerun()

        if run:
            if isinstance(category_value, str) and category_value.lower() == "all":
                category_value = None

            lines = [
                line.strip() for line in ingredients_input.split("\n") if line.strip()
            ]
            ingredients_query = "|".join(lines)

            with st.spinner("Recommendation in progress..."):
                try:
                    if recommendation_manager is not None:
                        recommendations = recommendation_manager.recommend(
                            input_value=ingredients_query,
                            input_type=config.input_mode.ingredients,
                            category_value=category_value,
                        )
                    else:
                        recommendations = remote_recommend(
                            input_value=ingredients_query,
                            input_type=config.input_mode.ingredients,
                            category_value=category_value,
                        )
                except Exception as e:
                    st.error(f"Error during recommendation: {e}")
                else:
                    st.session_state["last_recommendations"] = recommendations
                    st.session_state["last_report"] = None
        if st.session_state.get("last_recommendations") is not None:
            st.subheader("Summary of AI recommendations")
            _rec = st.session_state["last_recommendations"]
            if isinstance(_rec, (dict, list)):
                st.json(_rec)
            elif isinstance(_rec, str) and (
                "<br/>" in _rec or "<strong>" in _rec or "<p>" in _rec
            ):
                st.markdown(
                    _rec,
                    unsafe_allow_html=True,
                )
            else:
                st.write(_rec)
            if st.button("generate report", key="gen_report_ing"):
                with st.spinner("Report generation in progress..."):
                    try:
                        if report_manager is not None:
                            report = report_manager.generate(
                                recommendations=st.session_state[
                                    "last_recommendations"
                                ],
                            )
                        else:
                            report = remote_report(
                                recommendations=st.session_state[
                                    "last_recommendations"
                                ],
                            )
                    except Exception as e:
                        st.error(f"Error during report generation: {e}")
                    else:
                        st.session_state["last_report"] = report
        if st.session_state.get("last_report") is not None:
            st.subheader("Summary of AI report")
            st.write(st.session_state["last_report"])
    else:
        raise ValueError("Invalid input mode")
