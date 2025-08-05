import streamlit as st

def render_control(name, spec, df, allowed_cols, current_value, key_prefix="", help = None):
    """Render one UI control based on spec."""
    ctrl_type = spec.get("type")
    required = spec.get("required", False)
    options = []

    # Column-based controls
    if spec.get("allowed") == "col":
        options = allowed_cols["col"]
    elif spec.get("allowed") == "row":
        options = allowed_cols["row"]
    elif spec.get("allowed") == "hue":
        options = allowed_cols["hue"]
    elif spec.get("allowed") == "weight":
        options = allowed_cols["weight"]

    # Add 'None' option if not required
    if not required and ctrl_type in ["select", "multiselect"]:
        options = ["None"] + options

    # Default handling
    if current_value is None and not required:
        current_value = "None"

    widget_key = f"{key_prefix}_{name}"

    # Render different widget types
    if ctrl_type == "select":
        return st.selectbox(name, options, 
                            index=options.index(current_value) if current_value in options else 0,
                            key=widget_key,
                            help=help)
    
    elif ctrl_type == "multiselect":
        default = current_value if current_value else []
        # Normalize to list
        if not isinstance(default, list):
            default = [default]

        # Keep only defaults that are valid options
        default = [d for d in default if d in options]

        return st.multiselect(
            name,
            options,
            default=default,
            key=widget_key,
            help=help
        )

    
    
    
    
    
    elif ctrl_type == "number_input":
        min_val = spec.get("min")
        max_val = spec.get("max")
        default_val = spec.get("default")

        # Normalize None
        if current_value in (None, "None"):
            value = default_val
        else:
            value = current_value



        # safe cast to numbers
        try:
            if min_val is not None: min_val = int(min_val)
            if max_val is not None: max_val = int(max_val)
            if value is not None: value = int(value)
        except ValueError:
            st.error(f"Invalid number config for {name}")
            return None

        return st.number_input(
            label=name,
            min_value=min_val,
            max_value=max_val,
            value=value,
            key=f"{key_prefix}_{name}",
            help=help
        )
    elif ctrl_type == "range_slider":
        min_val = spec.get("min")
        max_val = spec.get("max")
        default_val = spec.get("default", [min_val, max_val])

        # Normalize None
        if current_value in (None, "None"):
            value = default_val
        else:
            value = current_value

        try:
            if min_val is not None: min_val = float(min_val)
            if max_val is not None: max_val = float(max_val)
            if value is not None: value = [float(value[0]), float(value[1])]
        except (ValueError, TypeError):
            st.error(f"Invalid range config for {name}")
            return None

        return st.slider(
            label=name,
            min_value=min_val,
            max_value=max_val,
            value=(value[0], value[1]),
            key=widget_key,
            help=help
        )

    elif ctrl_type == "segmented_control":
        options = spec.get("options", [])
        default = current_value if current_value else spec.get("default", options[0])
        return st.segmented_control(
            label=name,
            options=options,
            selection_mode="single",
            default=None,
            key=widget_key,
            help=help
        )



    elif ctrl_type == "checkbox":
        return st.checkbox(name, value=bool(current_value), key=widget_key, help=help)
    else:
        st.warning(f"Unknown control type {ctrl_type} for {name}")
        return current_value
