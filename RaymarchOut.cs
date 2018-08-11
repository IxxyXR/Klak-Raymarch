using UnityEngine;
using System.Reflection;
using Klak.Wiring;
using RaymarchingToolkit;


[AddComponentMenu("Klak/Wiring/Output/Component/Raymarch Out")]
public class RaymarchOut : NodeBase
{
    // Editable properties

    [SerializeField]
    RaymarchObject _targetComponent;

    [SerializeField]
    string _propertyName;
    
    // Node I/O

    [Inlet]
    public float radius {
        set {
            if (!enabled || _targetComponent == null) return;
            var prop = _targetComponent.shape.GetInput(_propertyName);
            prop.floatValue = value;
        }
    }

    
    // Private members

    private PropertyInfo _propertyInfo;

    void OnEnable()
    {
        if (_targetComponent != null)
            _propertyInfo = _targetComponent.GetType().GetProperty(_propertyName);
    }

}

