import React from 'react';
import axios from 'axios';

const ButtonHs = () => {
    const handleClick = async () => {
        try {
            await axios.post('http://127.0.0.1:5000/headpose');
            console.log('Script execution triggered.');
        } catch (error) {
            console.error('Error triggering script:', error);
        }
    };

    return (
        <button onClick={handleClick}>
            Head pose estimation
        </button>
    );
};

export default ButtonHs;
