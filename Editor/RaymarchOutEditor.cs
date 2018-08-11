using UnityEngine;
using UnityEditor;
using System;
using System.Linq;


[CustomEditor(typeof(RaymarchOut))]
public class RaymarchOutEditor : Editor
{
    SerializedProperty _targetComponent;
    SerializedProperty _propertyName;

    // Component list cache and its parent game object
    string[] _componentList;
    GameObject _cachedGameObject;

    // Check if a given component is capable of being a target.
    bool IsTargetable(Component component)
    {
        return component.GetType().GetProperty("enabled") != null;
    }

    // Cache component of a given game object if it's
    // different from a previously given game object.
    void CacheComponentList(GameObject gameObject)
    {
        if (_cachedGameObject == gameObject) return;

        _componentList = gameObject.GetComponents<Component>().
            Where(x => IsTargetable(x)).Select(x => x.GetType().Name).ToArray();

        _cachedGameObject = gameObject;
    }

    void OnEnable()
    {
        _targetComponent = serializedObject.FindProperty("_targetComponent");
        _propertyName = serializedObject.FindProperty("_propertyName");
    }

    void OnDisable()
    {
        _targetComponent = null;
        _componentList = null;
        _cachedGameObject = null;
        _propertyName = null;
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.PropertyField(_targetComponent);

        // Show the component selector when a component is given.
        if (_targetComponent.objectReferenceValue != null)
        {
            // Cache the component list.
            var component = (Component)_targetComponent.objectReferenceValue;
            CacheComponentList(component.gameObject);

            if (_componentList.Length > 0)
            {
                // Show the drop-down list.
                var index = Array.IndexOf(_componentList, component.GetType().Name);
                var newIndex = Mathf.Max(0, EditorGUILayout.Popup(" ", index, _componentList));

                // Update the component if the selection was changed.
                if (index != newIndex)
                    _targetComponent.objectReferenceValue = component.GetComponent(_componentList[newIndex]);
            }
            
            EditorGUILayout.PropertyField(_propertyName);
            EditorGUILayout.Space();
        }

        serializedObject.ApplyModifiedProperties();
    }
}

