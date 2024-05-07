import React from 'react';
import axios from 'axios';

const ButtonGe = () => {
    const handleClick = async () => {
        try {
            await axios.post('http://127.0.0.1:5000/gaze');
            console.log('Script execution triggered.');
        } catch (error) {
            console.error('Error triggering script:', error);
        }
    };

    return (
        <button onClick={handleClick}>
            Sleep estimation
        </button>
    );
};

export default ButtonGe;
